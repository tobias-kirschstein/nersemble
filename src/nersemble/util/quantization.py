from typing import Union

import numpy as np


def to_spherical(cartesian_points: np.ndarray) -> np.ndarray:
    x = cartesian_points[..., 0]
    y = cartesian_points[..., 1]
    z = cartesian_points[..., 2]

    radius = np.linalg.norm(cartesian_points, axis=-1, ord=2)
    theta = np.arctan2(np.sqrt(x * x + y * y), z)
    phi = np.arctan2(y, x)

    return np.stack([radius, theta, phi], axis=-1)


def to_cartesian(spherical_coordinates: np.ndarray) -> np.ndarray:
    radius = spherical_coordinates[..., 0]
    theta = spherical_coordinates[..., 1]
    phi = spherical_coordinates[..., 2]

    sin_theta = np.sin(theta)
    x = radius * np.cos(phi) * sin_theta
    y = radius * np.sin(phi) * sin_theta
    z = radius * np.cos(theta)

    return np.stack([x, y, z], axis=-1)


class Quantizer:

    def __init__(self,
                 min_values: Union[np.ndarray, float],
                 max_values: Union[np.ndarray, float],
                 bits: int,
                 mask_value: np.ndarray = 0,
                 separate_mask: bool = True):
        self._min_values = min_values
        self._max_values = max_values
        self._bits = bits
        self._mask_value = mask_value
        self._separate_mask = separate_mask

        self._mask_offset = 1 if separate_mask else 0  # Reserve bin 0 for mask
        self._n_buckets = 2 ** self._bits
        self._scale_factor = (self._n_buckets - 1 - self._mask_offset) / (self._max_values - self._min_values)

    def encode(self, values: np.ndarray) -> np.ndarray:
        mask = values != self._mask_value
        if len(mask.shape) > 2:
            mask = mask.any(axis=-1)

        scaled_values = (np.maximum(0,
                                    values - self._min_values) * self._scale_factor) + self._mask_offset  # Reserve bin 0 for mask
        assert scaled_values.min() >= self._mask_offset
        assert scaled_values.max() < self._n_buckets
        scaled_values[~mask] = 0
        quantized_values = scaled_values.round().astype(np.uint8 if self._bits == 8 else np.uint16)

        return quantized_values

    def decode(self, quantized_values: np.ndarray) -> np.ndarray:
        mask = quantized_values == self._mask_value
        if len(mask.shape) > 2:
            mask = mask.all(axis=-1)

        values = (quantized_values.astype(np.float32) - self._mask_offset) / self._scale_factor + self._min_values
        values[mask] = self._mask_value

        return values


class DepthQuantizer(Quantizer):

    def __init__(self, min_values: float = 0, max_values: float = 2, bits: int = 16, separate_mask: bool = True):
        super(DepthQuantizer, self).__init__(
            min_values=min_values,
            max_values=max_values,
            bits=bits,
            separate_mask=separate_mask)

    def encode(self, values: np.ndarray) -> np.ndarray:
        # Depth values > 2 are for sure outliers. Mask them
        values[values > self._max_values] = self._mask_value
        return super().encode(values)


class NormalsQuantizer(Quantizer):

    def __init__(self):
        super(NormalsQuantizer, self).__init__(
            min_values=np.array([0, 1 / 3 * np.pi, -np.pi]),
            max_values=np.array([1, np.pi, np.pi]),
            bits=8)

    def encode(self, values: np.ndarray) -> np.ndarray:
        mask = values != 0
        if len(mask.shape) > 2:
            mask = mask.any(axis=-1)

        spherical_normal_map = to_spherical(values)
        quantized_spherical_normal_map = super().encode(spherical_normal_map)
        quantized_spherical_normal_map[mask][..., 0] = 1  # Drop radius because it is always 1

        return quantized_spherical_normal_map

    def decode(self, quantized_values: np.ndarray) -> np.ndarray:
        mask = quantized_values != 0
        if len(mask.shape) > 2:
           mask = mask.any(axis=-1)

        spherical_normal_map = super().decode(quantized_values)
        normal_map = np.zeros_like(spherical_normal_map)
        normal_map[mask] = to_cartesian(spherical_normal_map[mask])

        return normal_map
