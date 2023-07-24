from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import writer
from torch.distributions import Normal
from torch_efficient_distloss import flatten_eff_distloss

from nersemble.nerfstudio.engine.generic_scheduler import GenericScheduler


@dataclass
class BaseModelConfig(ModelConfig):
    use_masked_rgb_loss: bool = False
    alpha_mask_threshold: float = 0.5  # alpha values above count as "foreground"

    lambda_alpha_loss: float = 0

    lambda_empty_loss: float = 0
    lambda_near_loss: float = 0
    lambda_depth_loss: float = 0

    eps_depth_initial: float = 0.9
    eps_depth_final: float = 0.01
    eps_depth_begin_step: int = 0
    eps_depth_end_step: int = 10000

    lambda_dist_loss: float = 0
    dist_loss_max_rays: int = 5000


class BaseModel(Model):
    config: BaseModelConfig

    def populate_modules(self):
        if self.config.lambda_empty_loss > 0 or self.config.lambda_near_loss > 0:
            self.sched_eps_depth = GenericScheduler(
                init_value=self.config.eps_depth_initial,
                final_value=self.config.eps_depth_final,
                begin_step=self.config.eps_depth_begin_step,
                end_step=self.config.eps_depth_end_step,
            )
        else:
            self.sched_eps_depth = None

    def get_depth_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) \
            -> List[TrainingCallback]:

        callbacks = super(BaseModel, self).get_training_callbacks(training_callback_attributes)

        # Window scheduling
        def update_scheduler_param(sched: GenericScheduler, name: str, step: int):
            sched.update(step)
            writer.put_scalar(name=f"scheduler_param/{name}", scalar=sched.get_value(), step=step)

        if self.sched_eps_depth is not None:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_scheduler_param,
                    args=[self.sched_eps_depth, "sched_eps_depth"],
                )
            )

        return callbacks

    def get_mask_per_ray(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if "mask" in batch:
            pixel_indices_per_ray = batch["local_indices"]  # [R, [c, y, x]]
            masks = batch["mask"].squeeze(3)  # [B, H, W]
            mask = masks[
                pixel_indices_per_ray[:, 0],
                pixel_indices_per_ray[:, 1],
                pixel_indices_per_ray[:, 2],
            ]

            return mask
        else:
            return None

    def get_alpha_per_ray(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        assert "alpha_map" in batch

        return batch["alpha_map"].squeeze(1) / 255.

    def get_masked_rgb_loss(self, batch: Dict[str, torch.Tensor], rgb_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch:
                should contain the GT image as 'image'.
                If it also contains a 'mask', the RGB loss will be only computed in the masked region.
            rgb_pred:
                the models predictions for the sampled rays

        Returns:
            the loss of the models RGB predictions vs the GT colors
        """

        image = batch["image"].to(self.device)

        if self.config.use_masked_rgb_loss and "mask" in batch:
            # Only compute RGB loss on non-masked pixels
            mask = self.get_mask_per_ray(batch)

            rgb_loss = self.rgb_loss(image[mask], rgb_pred[mask])
        elif self.config.use_masked_rgb_loss and "alpha_map" in batch:
            alpha_per_ray = self.get_alpha_per_ray(batch)
            mask = alpha_per_ray > self.config.alpha_mask_threshold
            rgb_loss = self.rgb_loss(image[mask], rgb_pred[mask])
            # rgb_loss = (alpha_per_ray.unsqueeze(1) * (image - rgb_pred) ** 2).sum() / alpha_per_ray.sum()
        else:
            rgb_loss = self.rgb_loss(image, rgb_pred)

        return rgb_loss

    def get_alpha_loss(self, batch: Dict[str, torch.Tensor], accumulation: torch.Tensor) -> Optional[torch.Tensor]:
        alpha_loss = None

        if self.config.lambda_alpha_loss is not None and self.config.lambda_alpha_loss > 0:

            accumulation_per_ray = accumulation.squeeze(1)  # [R]
            alpha_per_ray = self.get_alpha_per_ray(batch)
            # Only compute alpha loss in areas where the accumulation should be below 1
            idx_background = alpha_per_ray < 1

            if idx_background.any():
                alpha_loss = (accumulation_per_ray[idx_background] - alpha_per_ray[
                    idx_background]).abs().mean() * self.config.lambda_alpha_loss

        return alpha_loss

    def get_near_and_empty_loss(self,
                                batch: Dict[str, torch.Tensor],
                                ray_samples: RaySamples,
                                ray_indices: torch.Tensor,  # [S]
                                weights: torch.Tensor,  # [S, 1]
                                accumulation: torch.Tensor, ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        near_loss = None
        empty_loss = None

        if (self.config.lambda_empty_loss > 0 or self.config.lambda_near_loss > 0) and self.training:
            # Enforce space before depth to be empty
            eps = self.sched_eps_depth.value

            depth_targets_per_ray = batch["depth_maps"]  # [S]

            starts = ray_samples.frustums.starts.squeeze(1)  # [S]
            ends = ray_samples.frustums.ends.squeeze(1)  # [S]
            midpoints = (starts + ends) * 0.5
            target_depths_per_sample = depth_targets_per_ray[ray_indices]
            weights = weights.squeeze(1)  # [S]

            # Empty loss
            if self.config.lambda_empty_loss > 0:
                idx_very_near_samples = (target_depths_per_sample > 0) & (midpoints < target_depths_per_sample - eps)

                if idx_very_near_samples.any():
                    empty_loss = (weights[idx_very_near_samples] ** 2).mean()
                    empty_loss = self.config.lambda_empty_loss * empty_loss

            # Near loss
            if self.config.lambda_near_loss > 0:
                accumulation = accumulation.squeeze(1)  # [R]
                normal_distribution = Normal(0, (eps / 3) ** 2)

                idx_near_samples = (target_depths_per_sample > 0) \
                                   & (target_depths_per_sample - eps <= midpoints) \
                                   & (midpoints <= target_depths_per_sample + eps)

                expected_accumulation = normal_distribution.cdf(midpoints - target_depths_per_sample)

                if idx_near_samples.any():
                    loss_weight_map = torch.ones_like(ray_indices)

                    weights_cumsum = weights.cumsum(dim=0)  # [S]
                    idx_ray_change = torch.where(ray_indices != ray_indices.roll(1))[0]  # [R]
                    local_to_ray_idx = ray_indices.unique()  # The following code requires ray indices to be strictly ascending
                    ray_idx_to_local = torch.zeros(local_to_ray_idx.max() + 1, dtype=torch.long, device=weights.device)
                    ray_idx_to_local[local_to_ray_idx] = torch.arange(len(local_to_ray_idx), device=weights.device,
                                                                      dtype=torch.long)
                    local_ray_indices = ray_idx_to_local[ray_indices]
                    sample_idx_to_head = idx_ray_change[local_ray_indices]
                    head_cumsum = weights_cumsum[sample_idx_to_head]
                    head_weights = weights[sample_idx_to_head]
                    accumulated_weights = weights_cumsum - head_cumsum + head_weights  # [S]

                    difference = (accumulated_weights[idx_ray_change - 1].roll(-1) - accumulation[
                        local_to_ray_idx]).abs().max()

                    if not difference <= 1e-2:
                        print(
                            f"[WARNING!] Difference between accumulation and accumulated_weights!: {difference.item()}")

                    near_loss = (loss_weight_map[idx_near_samples] * (
                            accumulated_weights[idx_near_samples] - expected_accumulation[
                        idx_near_samples]) ** 2).mean()

                    near_loss = self.config.lambda_near_loss * near_loss

        return near_loss, empty_loss

    def get_depth_loss(self, batch: Dict[str, torch.Tensor], depths: torch.Tensor) -> Optional[torch.Tensor]:
        depth_loss = None
        if self.config.lambda_depth_loss > 0 and self.training:

            depth_targets_per_ray = batch["depth_maps"]
            predicted_depth_per_ray = depths.squeeze()  # [R]

            # Only use rays that actually hit a pixel where we have depth information
            depth_mask = depth_targets_per_ray > 0

            if depth_mask.any():
                depth_loss = ((
                                      depth_targets_per_ray[depth_mask] - predicted_depth_per_ray[
                                  depth_mask]) ** 2).mean()
                depth_loss *= self.config.lambda_depth_loss

        return depth_loss

    def get_dist_loss(self,
                      ray_samples: RaySamples,
                      ray_indices: torch.Tensor,  # [S]
                      weights: torch.Tensor,  # [S, 1]
                      ):
        dist_loss = None
        if self.config.lambda_dist_loss > 0:
            weights = weights.squeeze(1)  # [S]

            max_rays = self.config.dist_loss_max_rays
            # indices = (ray_indices.unsqueeze(1) == ray_indices.unique()[:max_rays]).any(dim=1)
            indices = ray_indices < max_rays

            ray_indices_small = ray_indices[indices]
            weights_small = weights[indices]
            ends = ray_samples.frustums.ends[indices].squeeze(-1)
            starts = ray_samples.frustums.starts[indices].squeeze(-1)

            midpoint_distances = (ends + starts) * 0.5
            intervals = ends - starts

            dist_loss = self.config.lambda_dist_loss * flatten_eff_distloss(
                weights_small, midpoint_distances, intervals, ray_indices_small
            )

        return dist_loss
