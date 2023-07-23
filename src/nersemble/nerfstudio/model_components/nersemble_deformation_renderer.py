from typing import Optional

import nerfacc
import torch
from nerfstudio.cameras.rays import RaySamples
from torch import nn


class DeformationRenderer(nn.Module):
    def forward(
            self,
            weights: torch.Tensor,
            ray_samples: RaySamples,
            ray_indices: Optional[torch.Tensor] = None,
            num_rays: Optional[int] = None,
    ) -> torch.Tensor:
        eps = 1e-10
        offsets = ray_samples.frustums.offsets

        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            deformation_per_ray = nerfacc.accumulate_along_rays(weights.squeeze(1),
                                                                values=offsets,
                                                                ray_indices=ray_indices,
                                                                n_rays=num_rays)
        else:
            deformation_per_ray = torch.sum(weights * offsets, dim=-2) / (torch.sum(weights, -2) + eps)

        return deformation_per_ray
