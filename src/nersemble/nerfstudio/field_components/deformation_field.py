from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components import MLP
from torch import nn

from nersemble.nerfstudio.field_components.windowed_nerf_encoding import WindowedNeRFEncoding
from nersemble.util.chunker import chunked
from nersemble.util.pytorch3d import se3_exp_map


@dataclass
class SE3DeformationFieldConfig:
    n_freq_pos = 7
    warp_code_dim: int = 8
    mlp_num_layers: int = 6
    mlp_layer_width: int = 128
    skip_connections: Tuple[int] = (4,)


def to_homogenous(v: torch.Tensor) -> torch.Tensor:
    return torch.concat([v, torch.ones_like(v[..., :1])], dim=-1)


def from_homogenous(v: torch.Tensor) -> torch.Tensor:
    return v[..., :3] / v[..., -1:]


class SE3WarpingField(nn.Module):

    def __init__(
            self,
            config: SE3DeformationFieldConfig,
    ) -> None:
        super().__init__()

        self.position_encoding = WindowedNeRFEncoding(
            in_dim=3,
            num_frequencies=config.n_freq_pos,
            min_freq_exp=0.0,
            max_freq_exp=config.n_freq_pos - 1,
            include_input=True
        )

        in_dim = self.position_encoding.get_out_dim() + config.warp_code_dim

        self.mlp_stem = MLP(
            in_dim=in_dim,
            out_dim=config.mlp_layer_width,
            num_layers=config.mlp_num_layers,
            layer_width=config.mlp_layer_width,
            skip_connections=config.skip_connections,
            out_activation=nn.ReLU(),
        )
        self.mlp_r = MLP(
            in_dim=config.mlp_layer_width,
            out_dim=3,
            num_layers=1,
            layer_width=config.mlp_layer_width,
        )
        self.mlp_v = MLP(
            in_dim=config.mlp_layer_width,
            out_dim=3,
            num_layers=1,
            layer_width=config.mlp_layer_width,
        )

        # diminish the last layer of SE3 Field to approximate an identity transformation
        nn.init.uniform_(self.mlp_r.layers[-1].weight, a=-1e-5, b=1e-5)
        nn.init.uniform_(self.mlp_v.layers[-1].weight, a=-1e-5, b=1e-5)
        nn.init.zeros_(self.mlp_r.layers[-1].bias)
        nn.init.zeros_(self.mlp_v.layers[-1].bias)

    def get_transform(self, positions: torch.Tensor,
                      warp_code: torch.Tensor,
                      windows_param: Optional[float] = None):
        encoded_xyz = self.position_encoding(
            positions,
            windows_param=windows_param,
        )  # (R, S, 3)

        feat = self.mlp_stem(torch.cat([encoded_xyz, warp_code], dim=-1))  # (R, S, D)

        r = self.mlp_r(feat).reshape(-1, 3)  # (R*S, 3)
        v = self.mlp_v(feat).reshape(-1, 3)  # (R*S, 3)

        screw_axis = torch.concat([v, r], dim=-1)  # (R*S, 6)
        screw_axis = screw_axis.to(positions.dtype)
        transforms = se3_exp_map(screw_axis)
        return transforms.permute(0, 2, 1)

    def apply_transform(self, positions: torch.Tensor, transforms: torch.Tensor):
        p = positions.reshape(-1, 3)

        warped_p = from_homogenous((transforms @ to_homogenous(p).unsqueeze(-1)).squeeze(-1))
        warped_p = warped_p.to(positions.dtype)

        idx_nan = warped_p.isnan()
        warped_p[idx_nan] = p[idx_nan]  # if deformation is NaN, just use original point

        # Reshape to shape of input positions tensor
        warped_p = warped_p.reshape(*positions.shape[: positions.ndim - 1], 3)

        return warped_p

    def forward(self, positions: torch.Tensor,
                warp_code: Optional[torch.Tensor] = None,
                windows_param: Optional[float] = None):
        if warp_code is None:
            return None

        transforms = self.get_transform(positions, warp_code, windows_param)
        return self.apply_transform(positions, transforms)


class SE3DeformationField(nn.Module):

    def __init__(self,
                 aabb: torch.Tensor,
                 deformation_field_config: SE3DeformationFieldConfig,
                 max_n_samples_per_batch: int = -1,
                 ):
        super(SE3DeformationField, self).__init__()

        # Parameter(..., requires_grad=False) ensures that AABB is moved to correct device
        self.aabb = nn.Parameter(aabb, requires_grad=False)

        self.se3_field = SE3WarpingField(deformation_field_config)
        self.max_n_samples_per_batch = max_n_samples_per_batch

    def forward(self,
                ray_samples: RaySamples,
                warp_code: Optional[torch.Tensor] = None,
                windows_param: Optional[float] = None) -> RaySamples:
        assert ray_samples.frustums.offsets is None or (
                ray_samples.frustums.offsets == 0).all(), "ray samples have already been warped"

        positions = ray_samples.frustums.get_positions()

        offsets = self.compute_offsets(positions, warp_code, windows_param)
        ray_samples.frustums.set_offsets(offsets)

        return ray_samples

    def compute_offsets(self,
                        positions: torch.Tensor,
                        warp_code: Optional[torch.Tensor] = None,
                        windows_param: Optional[float] = None):
        max_chunk_size = len(positions) if self.max_n_samples_per_batch == -1 else self.max_n_samples_per_batch

        offsets = []
        for positions_chunked, warp_code_chunked in chunked(max_chunk_size, positions, warp_code):
            positions_normalized = SceneBox.get_normalized_positions(positions_chunked, self.aabb)

            positions_warped = self.se3_field(positions_normalized,
                                              warp_code=warp_code_chunked,
                                              windows_param=windows_param)

            offsets_chunked = positions_warped - positions_normalized
            offsets.append(offsets_chunked)

        offsets = torch.cat(offsets, dim=0)
        return offsets
