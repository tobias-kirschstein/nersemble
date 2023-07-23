from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import Optional, Dict, List, Literal

import einops
import tinycudann as tcnn
import torch
from torch import nn


def posenc_window(windows_param: float, min_bands: float, max_bands: float, dim_encoding: int) -> torch.Tensor:
    """Windows a the encoding using a cosiney window.

    This is equivalent to taking a truncated Hann window and sliding it to the
    right along the frequency spectrum.

    Args:
        min_bands: the lower frequency band.
        max_bands: the upper frequency band.
        windows_param: will ease in each frequency as windows_param goes from 0.0 to num_freqs.

    Returns:
        A 1-d torch tensor with dim_encoding elements containing the window.
    """
    bands = torch.linspace(min_bands, max_bands, dim_encoding)
    x = torch.clamp(windows_param - bands, 0, 1)
    return 0.5 * (1 - torch.cos(torch.pi * x))


@dataclass
class TCNNHashEncodingConfig:
    n_dims_to_encode: int = 3  # Can be 3 or 4
    n_levels: int = 16
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 16
    per_level_scale: float = 1.4472692012786865
    interpolation: Literal['Linear', 'Nearest', 'Smoothstep'] = 'Linear'

    def setup(self, n_total_features: int) -> tcnn.Encoding:
        encoding = tcnn.Encoding(self.n_dims_to_encode, encoding_config={
            "otype": "HashGrid",
            "n_levels": self.n_levels,
            "n_features_per_level": 8 if n_total_features >= 8 else n_total_features,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale,
            "interpolation": self.interpolation
        })

        return encoding


@dataclass
class HashEnsembleConfig:
    n_hash_encodings: int
    """ Number of different hash encodings """
    hash_encoding_config: TCNNHashEncodingConfig
    """ config for each individual hash encoding """
    disable_initial_hash_ensemble: bool = False
    """ If set, first hash encoding will have fixed blend weight=1 until the remaining ones start to fade in """
    use_soft_transition: bool = False
    """ If set, the fixed weight=1 of the first hash encoding will gradually be blended with the actual
    learnable weight while window_param in [1-2]. 
    This should avoid a hard jump of weight=1 to weight=beta_1 when the second hash encoding starts to fade in."""


class HashEnsemble(nn.Module):

    def __init__(self, config: HashEnsembleConfig):
        super(HashEnsemble, self).__init__()

        self.n_hash_encodings = config.n_hash_encodings
        self.hash_encoding_config = config.hash_encoding_config
        self.disable_initial_hash_ensemble = config.disable_initial_hash_ensemble
        self.use_soft_transition = config.use_soft_transition

        n_total_features = config.n_hash_encodings * config.hash_encoding_config.n_features_per_level
        assert n_total_features <= 8 \
               or n_total_features % 8 == 0, \
            "Number of features in hashtables must either be smaller than 8 or a multiple of 8!"
        self.hash_encodings = []
        for i_hash_encoding in range(ceil(n_total_features / 8)):
            self.hash_encodings.append(config.hash_encoding_config.setup(n_total_features))

        self.hash_encodings = nn.ModuleList(self.hash_encodings)

        dim_hash_encoding = config.hash_encoding_config.n_levels * config.hash_encoding_config.n_features_per_level

        self.n_output_dims = dim_hash_encoding

    def forward(self,
                in_tensor: torch.Tensor,
                conditioning_code: torch.Tensor,
                windows_param: Optional[float] = None,
                window_hash_encodings: Optional[float] = None) -> torch.Tensor:

        B = in_tensor.shape[0]

        embeddings = []
        for h, hash_encoding in enumerate(self.hash_encodings):
            embedding = hash_encoding(in_tensor)
            embeddings.append(embedding)

        embeddings = torch.stack(embeddings, dim=1)  # [B, C, 8 * L]
        C = embeddings.shape[1]
        L = self.hash_encoding_config.n_levels
        F = self.hash_encoding_config.n_features_per_level
        P = int(8 / F) if F * self.n_hash_encodings >= 8 else self.n_hash_encodings

        embeddings = einops.rearrange(embeddings, 'b c (l p f) -> b (l f) (c p) ', l=L, p=P, f=F)

        # embeddings = embeddings.reshape((B, C, L, P, F))
        # embeddings = embeddings.transpose(2, 3)  # [B, C, P, L, F]
        # embeddings = embeddings.reshape((B, C*P, L*F))
        # embeddings = embeddings.transpose(1, 2)  # [B, D, H]

        if window_hash_encodings is not None:
            # Gradually add more tables

            if window_hash_encodings == 1 and self.disable_initial_hash_ensemble:
                # Force deformation network to learn correspondences as long as only one table is active
                conditioning_code = torch.ones_like(conditioning_code)
            elif self.use_soft_transition and window_hash_encodings < 2:
                # Slowly migrate to using the actual conditioning code instead of fixing the blend weights to 1
                alpha = window_hash_encodings - 1  # Goes from 0 -> 1

                # Only first entry of conditioning code is responsible for first table
                conditioning_code = alpha * conditioning_code
                conditioning_code[:, 0] += (1 - alpha) * 1

            window = posenc_window(window_hash_encodings,
                                   0,
                                   self.n_hash_encodings - 1,
                                   self.n_hash_encodings)  # [H]
            window = window.unsqueeze(0).unsqueeze(1).to(embeddings)  # [1, 1, H]
            embeddings = window * embeddings

        # TODO: Probably wasn't used and can be removed
        if windows_param is not None:
            # Gradually add higher frequency detail
            window = posenc_window(windows_param,
                                   0,
                                   self.hash_encoding_config.n_levels - 1,
                                   self.hash_encoding_config.n_levels)  # [L]
            window = window.repeat_interleave(self.hash_encoding_config.n_features_per_level)  # [L*F = D]
            window = window.unsqueeze(0).unsqueeze(2).to(embeddings)  # [1, D, 1]
            embeddings = window * embeddings

        assert conditioning_code.shape[-1] == self.n_hash_encodings, \
            "If blend mixing type is chosen, conditioning code needs to have as many dimensions as there are " \
            "hashtables in the encoding"

        conditioning_code = conditioning_code.to(embeddings)  # Make conditioning code half precision
        blended_embeddings = torch.einsum('bdh,bh->bd', embeddings, conditioning_code)

        return blended_embeddings

    def get_out_dim(self) -> int:
        return self.n_output_dims

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        param_groups = defaultdict(list)

        param_groups["fields"] = list(self.hash_encodings.parameters())

        return param_groups
