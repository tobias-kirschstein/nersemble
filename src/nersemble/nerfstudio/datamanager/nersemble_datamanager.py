from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Type

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager, TDataset
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader

from nersemble.nerfstudio.data.nersemble_pixel_sampler import NeRSemblePixelSampler
from nersemble.nerfstudio.dataparser.nersemble_dataparser import NeRSembleDataParserOutputs
from nersemble.nerfstudio.dataset.nersemble_dataset import NeRSembleInputDataset

# These keys will additionally be passed to the model
# They need separate handling since they refer to information that is relevant "per image" and not "per pixel/ray"
ADDITIONAL_METADATA = ["depth_map", "timesteps", "cam_ids"]


@dataclass
class NeRSembleVanillaDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: NeRSembleVanillaDataManager)

    max_cached_items: int = -1  # How many images will at most be held in RAM
    use_cache_compression: bool = False  # Stores images as uint8 instead of float in RAM to save space (May be lossy!)


class NeRSembleVanillaDataManager(VanillaDataManager):
    config: NeRSembleVanillaDataManagerConfig
    train_dataparser_outputs: NeRSembleDataParserOutputs

    def setup_train(self):
        super().setup_train()

        # For logging train images during evaluation as well
        self.train_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def create_train_dataset(self) -> NeRSembleInputDataset:
        """Sets up the data loaders for training"""
        train_dataset = NeRSembleInputDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            max_cached_items=self.config.max_cached_items,
            use_cache_compression=self.config.use_cache_compression
        )
        # Communicate camera frustums to model
        train_dataset.metadata["camera_frustums"] = self.train_dataparser_outputs.camera_frustums
        return train_dataset

    def create_eval_dataset(self) -> NeRSembleInputDataset:
        """Sets up the data loaders for evaluation"""
        return NeRSembleInputDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(self, dataset: TDataset, *args: Any, **kwargs: Any) -> NeRSemblePixelSampler:
        """Use FamudyPixelSampler instead, which has special handling for per-image metadata such as
        timestap, participant_id"""
        if self.config.patch_size > 1:
            raise NotImplementedError()
            return PatchPixelSampler(*args, **kwargs, patch_size=self.config.patch_size)

        return NeRSemblePixelSampler(additional_metadata=ADDITIONAL_METADATA, *args, **kwargs)

    def _add_metadata_to_ray_bundle(self, ray_bundle: RayBundle, batch: Dict):
        for metadata_key in ADDITIONAL_METADATA:
            if metadata_key in batch:
                metadata = batch[metadata_key]
                if not isinstance(metadata, torch.Tensor):
                    metadata = torch.tensor(batch[metadata_key]).repeat(*ray_bundle.shape).to(ray_bundle.origins.device)
                ray_bundle.metadata[metadata_key] = metadata.unsqueeze(-1)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_train(step)

        self._add_metadata_to_ray_bundle(ray_bundle, batch)

        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_eval(step)

        self._add_metadata_to_ray_bundle(ray_bundle, batch)

        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        image_idx, ray_bundle, batch = super().next_eval_image(step)

        self._add_metadata_to_ray_bundle(ray_bundle, batch)

        return image_idx, ray_bundle, batch

    def next_train_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.train_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])

            self._add_metadata_to_ray_bundle(camera_ray_bundle, batch)

            return image_idx, camera_ray_bundle, batch

        raise ValueError("No more train images")
