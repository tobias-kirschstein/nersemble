from typing import Dict, Optional, List

import torch
from nerfstudio.data.pixel_samplers import PixelSampler


class NeRSemblePixelSampler(PixelSampler):

    def __init__(self,
                 num_rays_per_batch: int,
                 keep_full_image: bool = False,
                 additional_metadata: Optional[List[str]] = None,
                 **kwargs) -> None:
        super().__init__(num_rays_per_batch, keep_full_image, **kwargs)

        self._per_image_attributes = ["image_idx"] # nerfstudio default
        if additional_metadata is not None:
            self._per_image_attributes.extend(additional_metadata)
            self._additional_metadata = additional_metadata
        else:
            self._additional_metadata = list()

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Extends nerfstudio Pixelsampler to treat per-image metadata (e.g., timestep, participant_id) the same way
        as the image_idx.

        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask" in batch:
            indices = self.sample_method(
                num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
            )
        else:
            indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()

        collated_batch = {
            key: value[c, y, x] for key, value in batch.items()
            if key not in self._per_image_attributes and value is not None
        }

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        for per_image_attribute in self._additional_metadata:
            if per_image_attribute in batch:
                collated_batch[per_image_attribute] = batch[per_image_attribute][c]

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]  # NB: This line changes c!!!
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch
