from typing import Dict

import numpy as np
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from numpy.typing import NDArray
from PIL import Image

from nersemble.nerfstudio.dataparser.nersemble_dataparser import NeRSembleDataParserOutputs


class InMemoryInputDataset(InputDataset):
    """
    Can be used as a drop-in replacement for InputDataset.
    Instead of loading the images everytime upon request, it caches them.
    This will increase memory consumption over time until all images have been loaded once.
    """

    def __init__(self,
                 dataparser_outputs: DataparserOutputs,
                 scale_factor: float = 1.0,
                 max_cached_items: int = -1,
                 use_cache_compression: bool = False):
        super(InMemoryInputDataset, self).__init__(dataparser_outputs, scale_factor=scale_factor)

        self._cached_items = dict()
        self._max_cached_items = max_cached_items
        self._use_cache_compression = use_cache_compression

    def _compress(self, item: Dict) -> Dict:
        if not self._use_cache_compression:
            return item

        item = item.copy()
        # Only store uint8 values for every pixel channel (discretization introduces lossy compression!)
        item['image'] = (item['image'] * 255).round().to(torch.uint8)
        return item

    def _uncompress(self, item: Dict) -> Dict:
        if not self._use_cache_compression:
            return item

        item = item.copy()
        # Cast image back into float
        item['image'] = item['image'].float() / 255.
        return item

    def __getitem__(self, image_idx):
        if image_idx in self._cached_items:
            item = self._uncompress(self._cached_items[image_idx])
        else:
            item = super().__getitem__(image_idx)
            if self._max_cached_items == -1 or len(self._cached_items) < self._max_cached_items:
                # Only cache item if number of cached items hasn't been exceeded yet
                self._cached_items[image_idx] = self._compress(item)

        return item


class NeRSembleInputDataset(InMemoryInputDataset):
    _dataparser_outputs: NeRSembleDataParserOutputs

    def get_numpy_image(self, image_idx: int) -> NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)

        # Apply color correction after loading images
        if self._dataparser_outputs.color_correction_filenames is not None:
            image = np.array(pil_image, dtype="float") / 255  # shape is (h, w) or (h, w, 3 or 4)
            if len(image.shape) == 2:
                image = image[:, :, None].repeat(3, axis=2)

            has_alpha_channels = False
            if image.shape[-1] == 4:
                alpha_channels = image[:, :, 3]
                image = image[:, :, :3]
                has_alpha_channels = True
            affine_color_transform = np.load(self._dataparser_outputs.color_correction_filenames[image_idx])
            image = image @ affine_color_transform[:3, :3] + affine_color_transform[np.newaxis, :3, 3]
            image = np.clip(image, 0, 1)
            if has_alpha_channels:
                image = np.concatenate([image, alpha_channels[:, :, np.newaxis]], axis=-1)
            image = (image * 255).astype(np.uint8)
        else:
            image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
            if len(image.shape) == 2:
                image = image[:, :, None].repeat(3, axis=2)

        # Add alpha channel to loaded images
        if self._dataparser_outputs.alpha_channel_filenames is not None:
            alpha_channel_filename = self._dataparser_outputs.alpha_channel_filenames[image_idx]
            pil_alpha_image = Image.open(alpha_channel_filename)
            pil_alpha_image = pil_alpha_image.resize(pil_image.size, resample=Image.BILINEAR)

            alpha_image = np.asarray(pil_alpha_image, dtype="uint8")
            image = np.concatenate([image, alpha_image[..., None]], axis=-1)

        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_metadata(self, data: Dict) -> Dict:
        metadata = dict()
        image_idx = data['image_idx']
        for key, data_func_dict in self.metadata.items():
            if key == 'camera_frustums':
                # Camera frustums are stored in metadata, but are not typical per-image metadata as the other fields
                continue

            assert "func" in data_func_dict, "Missing function to process data: specify `func` in `additional_inputs`"
            func = data_func_dict["func"]
            assert "kwargs" in data_func_dict, "No data to process: specify `kwargs` in `additional_inputs`"
            metadata.update(func(image_idx, **data_func_dict["kwargs"]))

        del data

        return metadata
