from __future__ import annotations

from dataclasses import field, dataclass
from math import sqrt
from typing import Type, Dict, List, Optional, Tuple

import nerfacc
import torch
from dreifus.util.colormap import apply_scene_flow_colormap
from jaxtyping import Shaped
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.instant_ngp import NGPModel, InstantNGPModelConfig
from nerfstudio.utils import writer, colormaps
from nerfstudio.utils.colormaps import ColormapOptions
from torch import Tensor, nn
from torch.nn import Parameter, init
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nersemble.nerfstudio.engine.generic_scheduler import GenericScheduler
from nersemble.nerfstudio.field_components.deformation_field import SE3DeformationField, SE3DeformationFieldConfig
from nersemble.nerfstudio.field_components.hash_ensemble import HashEnsembleConfig
from nersemble.nerfstudio.fields.nersemble_nerfacto_field import NeRSembleNeRFactoField
from nersemble.nerfstudio.model_components.nersemble_deformation_renderer import DeformationRenderer
from nersemble.nerfstudio.model_components.nersemble_volumetric_sampler import NeRSembleVolumetricSampler
from nersemble.nerfstudio.models.base import BaseModel, BaseModelConfig


@dataclass
class NeRSembleNGPModelConfig(InstantNGPModelConfig, BaseModelConfig):
    # class NeRSembleNGPModelConfig(InstantNGPModelConfig):
    _target: Type = field(default_factory=lambda: NeRSembleNGPModel)

    n_timesteps: int = 1
    latent_dim_time: int = 128
    spherical_harmonics_degree: int = 0

    # Hash Ensemble
    use_hash_ensemble: bool = False
    hash_ensemble_config: Optional[HashEnsembleConfig] = None

    # Deformation Field
    use_deformation_field: bool = False
    deformation_field_config: Optional[SE3DeformationFieldConfig] = None
    use_separate_deformation_time_embedding: bool = True
    # If false, deformation field time embedding needs to be the same as hash ensemble time embedding

    # Window scheduler
    window_deform_begin: int = 0
    window_deform_end: int = 0
    window_hash_encodings_begin: int = 0
    window_hash_encodings_end: int = 1

    # Ray marching
    early_stop_eps: float = 1e-4
    occ_thre: float = 1e-2
    disable_occupancy_grid: bool = False  # If set, occupancy grid is just plain 1s everywhere
    occupancy_grid_ema_decay: float = 0.95
    occupancy_grid_warmup_steps: int = 256
    max_n_samples_per_batch: int = -1  # If the number of ray samples exceeds this value, they will be put in chunks through the model. Smaller values lower speed, but reduce GPU memory consumption


    # View Frustum Culling
    use_view_frustum_culling: bool = False
    view_frustum_culling: int = 2  # Intersection of that many cameras defines subset of 3D space


class NeRSembleNGPModel(NGPModel, BaseModel):
    config: NeRSembleNGPModelConfig

    def populate_modules(self):
        """
        NeRSemble overwrites the default
         - pass spherical_harmonics_degree

        """
        super(NGPModel, self).populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = NeRSembleNeRFactoField(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            spatial_distortion=scene_contraction,

            # NeRSemble additions
            spherical_harmonics_degree=self.config.spherical_harmonics_degree,
            use_hash_ensemble=self.config.use_hash_ensemble,
            hash_ensemble_config=self.config.hash_ensemble_config,
            use_appearance_embedding=self.config.use_appearance_embedding,
            max_n_samples_per_batch=self.config.max_n_samples_per_batch
        )

        # Deformation Field
        self.deformation_field = None
        if self.config.use_deformation_field:
            self.deformation_field = SE3DeformationField(self.scene_box.aabb, self.config.deformation_field_config,
                                                         max_n_samples_per_batch=self.config.max_n_samples_per_batch)

        # Time embedding
        self.time_embedding = None
        if self.config.use_deformation_field or self.config.use_hash_ensemble:
            self.time_embedding = nn.Embedding(self.config.n_timesteps, self.config.latent_dim_time)
            init.normal_(self.time_embedding.weight, mean=0., std=0.01 / sqrt(self.config.latent_dim_time))

            if self.config.use_separate_deformation_time_embedding:
                self.time_embedding_deformation = nn.Embedding(self.config.n_timesteps,
                                                               self.config.deformation_field_config.warp_code_dim)
                init.normal_(self.time_embedding_deformation.weight, mean=0.,
                             std=0.01 / sqrt(self.config.deformation_field_config.warp_code_dim))

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 1000
        # Occupancy Grid.
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # Sampler
        self.sampler = NeRSembleVolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field_density_fn,
            scene_aabb=self.scene_box.aabb,
            camera_frustums=self.kwargs['metadata']['camera_frustums'],
            view_frustum_culling=self.config.view_frustum_culling if self.config.use_view_frustum_culling else None
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_deformation = DeformationRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Scheduler
        self.sched_window_deform = None
        if self.config.window_deform_end >= 1:
            self.sched_window_deform = GenericScheduler(
                init_value=0,
                final_value=self.config.deformation_field_config.n_freq_pos,
                begin_step=self.config.window_deform_begin,
                end_step=self.config.window_deform_end,
            )

        self.sched_window_hash_encodings = None
        if self.config.use_hash_ensemble and self.config.window_hash_encodings_end > 0:
            self.sched_window_hash_encodings = GenericScheduler(
                init_value=1,
                final_value=self.config.hash_ensemble_config.n_hash_encodings,
                begin_step=self.config.window_hash_encodings_begin,
                end_step=self.config.window_hash_encodings_end,
            )

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[
        TrainingCallback]:

        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field_density_fn(
                    x,
                    torch.randint(0, self.config.n_timesteps, (x.shape[0], 1), dtype=torch.int, device=x.device) / (
                            self.config.n_timesteps - 1)
                ) * self.config.render_step_size,
                n=16,
                occ_thre=self.config.occ_thre,
                ema_decay=self.config.occupancy_grid_ema_decay,
                warmup_steps=self.config.occupancy_grid_warmup_steps
            )

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

        # Window scheduling
        def update_window_param(sched: GenericScheduler, name: str, step: int):
            sched.update(step)
            writer.put_scalar(name=f"window_param/{name}", scalar=sched.get_value(), step=step)

        if self.sched_window_deform is not None:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_window_param,
                    args=[self.sched_window_deform, "sched_window_deform"],
                )
            )

        if self.sched_window_hash_encodings is not None:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_window_param,
                    args=[self.sched_window_hash_encodings, "sched_window_hash_encodings"],
                )
            )

        # callbacks = super(NeRSembleNGPModel, self).get_training_callbacks(training_callback_attributes)
        depth_training_callbacks = self.get_depth_training_callbacks(training_callback_attributes)
        return callbacks + depth_training_callbacks

    def field_density_fn(self,
                         positions: Shaped[Tensor, "*bs 3"],
                         times: Optional[Shaped[Tensor, "*bs 1"]]) -> Shaped[Tensor, "*bs 1"]:

        if self.config.disable_occupancy_grid:
            return torch.ones((positions.shape[0],), dtype=positions.dtype, device=positions.device)

        window_hash_encodings = self.sched_window_hash_encodings.value if self.sched_window_hash_encodings is not None else None
        window_deform = self.sched_window_deform.value if self.sched_window_deform is not None else None

        time_codes = None
        time_codes_deformation = None
        if self.time_embedding is not None:
            assert times is not None, "Times need to be provided to NeRSemble's density_fn"
            timesteps = (times * (self.config.n_timesteps - 1)).round().int().squeeze(1)
            time_codes = self.time_embedding(timesteps)

            if self.config.use_separate_deformation_time_embedding:
                time_codes_deformation = self.time_embedding_deformation(timesteps)
            else:
                time_codes_deformation = time_codes

        if self.config.use_deformation_field:
            offsets = self.deformation_field.compute_offsets(positions, time_codes_deformation, window_deform)
            positions = positions + offsets

        densities = self.field.density_fn(positions,
                                     times,
                                     window_hash_encodings=window_hash_encodings,
                                     time_codes=time_codes)

        return densities

    def warp_ray_samples(self, ray_samples: RaySamples, time_codes: Optional[torch.Tensor] = None) -> RaySamples:
        window_deform = self.sched_window_deform.value if self.sched_window_deform is not None else None

        if self.deformation_field is not None:
            # Initialize all offsets with 0
            assert ray_samples.frustums.offsets is None, "ray samples have already been warped"

            # Deform all samples into the latent canonical space
            self.deformation_field(ray_samples, warp_code=time_codes, windows_param=window_deform)

        return ray_samples

    def get_outputs(self, ray_bundle: RayBundle):
        window_hash_encodings = self.sched_window_hash_encodings.value if self.sched_window_hash_encodings is not None else None

        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
                early_stop_eps=self.config.early_stop_eps
            )

        if ray_samples.metadata is None:
            ray_samples.metadata = dict()

        # Lookup time codes via timesteps if they were not already provided
        if ray_bundle.times is not None:
            # Obtain timesteps from "times" field of ray bundle. [0-1] -> [0 - (T-1)]
            timesteps = (ray_bundle.times[ray_indices] * (self.config.n_timesteps - 1)).round().int().squeeze(1)
        elif 'timesteps' in ray_bundle.metadata:
            timesteps = ray_bundle.metadata['timesteps'][ray_indices].squeeze(1)

        time_codes_deformation = None
        if self.time_embedding is not None:
            if 'time_codes' not in ray_samples.metadata:
                time_codes = self.time_embedding(timesteps)
                ray_samples.metadata['time_codes'] = time_codes
                # TODO: Maybe, we do not put the time_codes into ray_samples?
                #   Or put time_codes_deformation in there as well?

            if self.config.use_separate_deformation_time_embedding:
                time_codes_deformation = self.time_embedding_deformation(timesteps)
            else:
                time_codes_deformation = ray_samples.metadata['time_codes']

        ray_samples = self.warp_ray_samples(ray_samples, time_codes_deformation)

        field_outputs = self.field(ray_samples, window_hash_encodings=window_hash_encodings)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],

            # Use single-element tuples for additional data that is not per-ray
            # This avoids errors in get_outputs_for_camera_ray_bundle() which tries to create image-dimension outputs
            # for all fields in the outputs dict
            "ray_samples": (ray_samples,),
            "ray_indices": (ray_indices,),
            "weights": (weights,)
        }

        if ray_samples.frustums.offsets is not None:
            deformation_per_ray = self.renderer_deformation(weights=weights, ray_samples=ray_samples,
                                                            ray_indices=ray_indices, num_rays=num_rays)
            outputs["deformation"] = deformation_per_ray

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = dict()

        accumulation = outputs["accumulation"]
        depths = outputs["depth"]
        ray_samples = outputs["ray_samples"][0]
        ray_indices = outputs["ray_indices"][0]
        weights = outputs["weights"][0]

        rgb_loss = self.get_masked_rgb_loss(batch, outputs["rgb"])
        loss_dict["rgb_loss"] = rgb_loss

        # Alpha supervision
        alpha_loss = self.get_alpha_loss(batch, accumulation)
        if alpha_loss is not None:
            loss_dict["alpha_loss"] = alpha_loss

        # Depth supervision
        near_loss, empty_loss = self.get_near_and_empty_loss(batch,
                                                             ray_samples,
                                                             ray_indices,
                                                             weights,
                                                             accumulation)

        depth_loss = self.get_depth_loss(batch, depths)

        if near_loss is not None:
            loss_dict["near_loss"] = near_loss

        if empty_loss is not None:
            loss_dict["empty_loss"] = empty_loss

        if depth_loss is not None:
            loss_dict["depth_loss"] = depth_loss

        # Distortion loss
        dist_loss = self.get_dist_loss(ray_samples, ray_indices, weights)

        if dist_loss is not None:
            loss_dict["dist_loss"] = dist_loss

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        rgb = outputs["rgb"]
        image = batch["image"].to(self.device)

        metrics_dict = dict()

        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()

        if "alpha_map" in batch:
            mask = batch["alpha_map"].squeeze(1) > 127
            metrics_dict["psnr_masked"] = self.psnr(rgb[mask], image[mask])

        return metrics_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        error_image = ((rgb - image) ** 2).mean(dim=-1).unsqueeze(-1)
        error_image = colormaps.apply_colormap(error_image, colormap_options=ColormapOptions(colormap="turbo"))

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)
        mse = self.rgb_loss(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim),
            "lpips": float(lpips),
            "mse": float(mse),
            "cam_id": float(batch['cam_ids']),
            # cam_ids are necessary for correct image logging in trainer, but has to be float
        }  # type: ignore

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "error": error_image
        }

        if "deformation" in outputs:
            deformation_img = apply_scene_flow_colormap(outputs["deformation"])
            images_dict["deformation"] = deformation_img

        # Metrics and images with alpha masks
        if "alpha_map" in batch:
            alpha_mask = batch["alpha_map"] / 255.  # [H, W, 1]
            alpha_mask = torch.from_numpy(alpha_mask).to(rgb)

            image_masked = batch["image"].clone().to(self.device)
            rgb_masked = rgb[0].permute(1, 2, 0)

            image_masked = alpha_mask * image_masked + (1 - alpha_mask)
            rgb_masked = alpha_mask * rgb_masked + (1 - alpha_mask)

            combined_rgb_masked = torch.cat([image_masked, rgb_masked], dim=1)

            image_masked = torch.moveaxis(image_masked, -1, 0)[None, ...]
            rgb_masked = torch.moveaxis(rgb_masked, -1, 0)[None, ...]

            psnr_masked = self.psnr(image_masked, rgb_masked)
            ssim_masked = self.ssim(image_masked, rgb_masked)
            lpips_masked = self.lpips(image_masked, rgb_masked)
            mse_masked = self.rgb_loss(image_masked, rgb_masked)

            metrics_dict["psnr_masked"] = float(psnr_masked)
            metrics_dict["ssim_masked"] = float(ssim_masked)
            metrics_dict["lpips_masked"] = float(lpips_masked)
            metrics_dict["mse_masked"] = float(mse_masked)

            images_dict["img_masked"] = combined_rgb_masked

        return metrics_dict, images_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()

        if self.time_embedding is not None:
            param_groups["embeddings"] = list(self.time_embedding.parameters())

            if self.config.use_separate_deformation_time_embedding:
                param_groups["embeddings"].extend(list(self.time_embedding_deformation.parameters()))

        if self.config.use_deformation_field:
            param_groups["deformation_field"] = list(self.deformation_field.parameters())

        return param_groups


