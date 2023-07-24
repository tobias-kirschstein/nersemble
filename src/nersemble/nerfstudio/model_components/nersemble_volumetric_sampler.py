from typing import Optional, Tuple, List

import torch
from jaxtyping import Float
from nerfacc import OccGridEstimator
from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.model_components.ray_samplers import VolumetricSampler, DensityFn
from torch import Tensor

from nersemble.nerfstudio.model_components.frustum import TorchFrustum


class NeRSembleVolumetricSampler(VolumetricSampler):



    def __init__(self,
                 occupancy_grid: OccGridEstimator,
                 density_fn: Optional[DensityFn] = None,
                 scene_aabb: Optional[torch.Tensor] = None,
                 camera_frustums: Optional[List[TorchFrustum]] = None,
                 view_frustum_culling: Optional[int] = None):
        super().__init__(occupancy_grid, density_fn)

        self.camera_frustums = camera_frustums
        self.view_frustum_culling = view_frustum_culling

        if camera_frustums is not None and view_frustum_culling is not None:
            camera_frustum_grid_resolution = self.occupancy_grid.resolution
            grid_xs, grid_ys, grid_zs = torch.meshgrid(
                torch.linspace(scene_aabb[0][0], scene_aabb[1][0], steps=camera_frustum_grid_resolution[0]),
                torch.linspace(scene_aabb[0][1], scene_aabb[1][1], steps=camera_frustum_grid_resolution[1]),
                torch.linspace(scene_aabb[0][2], scene_aabb[1][2], steps=camera_frustum_grid_resolution[2])
            )
            grid_points = torch.stack([grid_xs, grid_ys, grid_zs], dim=-1).view(-1, 3).to(camera_frustums[0]._half_space_collection.offsets.device)
            visibility_masks = [camera_frustum.contains_points(grid_points) for camera_frustum in
                                self.camera_frustums]
            visibility_mask = torch.stack(visibility_masks).sum(dim=0) >= self.view_frustum_culling
            visibility_mask = visibility_mask.view(*camera_frustum_grid_resolution)
            self.camera_frustum_grid = visibility_mask
        else:
            self.camera_frustum_grid = None

    def forward(self,
                ray_bundle: RayBundle,
                render_step_size: float,
                near_plane: float = 0.0,
                far_plane: Optional[float] = None,
                alpha_thre: float = 0.01,
                cone_angle: float = 0.0,
                early_stop_eps: float = 1e-4) -> Tuple[
        RaySamples, Float[Tensor, "total_samples "]]:
        """Generate ray samples in a bounding box.

                Args:
                    ray_bundle: Rays to generate samples for
                    render_step_size: Minimum step size to use for rendering
                    near_plane: Near plane for raymarching
                    far_plane: Far plane for raymarching
                    alpha_thre: Opacity threshold skipping samples.
                    cone_angle: Cone angle for raymarching, set to 0 for uniform marching.
                    early_stop_eps: Threshold for skipping invisible space.

                Returns:
                    a tuple of (ray_samples, packed_info, ray_indices)
                    The ray_samples are packed, only storing the valid samples.
                    The ray_indices contains the indices of the rays that each sample belongs to.
                """

        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        times = ray_bundle.times

        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            t_min = ray_bundle.nears.contiguous().reshape(-1)
            t_max = ray_bundle.fars.contiguous().reshape(-1)

        else:
            t_min = None
            t_max = None

        if far_plane is None:
            far_plane = 1e10

        if ray_bundle.camera_indices is not None:
            camera_indices = ray_bundle.camera_indices.contiguous()
        else:
            camera_indices = None

        if self.camera_frustum_grid is not None:
            # Intersect occupancy grid with view frustums
            # For now, we can only use view frustum culling for the coarsest level
            self.occupancy_grid.binaries[0] = self.occupancy_grid.binaries[0] & self.camera_frustum_grid

        ray_indices, starts, ends = self.occupancy_grid.sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=t_min,
            t_max=t_max,
            sigma_fn=self.get_sigma_fn(rays_o, rays_d, times),
            render_step_size=render_step_size,
            near_plane=near_plane,
            far_plane=far_plane,
            stratified=self.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
            early_stop_eps=early_stop_eps,
        )
        num_samples = starts.shape[0]
        if num_samples == 0:
            # create a single fake sample and update packed_info accordingly
            # this says the last ray in packed_info has 1 sample, which starts and ends at 1
            ray_indices = torch.zeros((1,), dtype=torch.long, device=rays_o.device)
            starts = torch.ones((1,), dtype=starts.dtype, device=rays_o.device)
            ends = torch.ones((1,), dtype=ends.dtype, device=rays_o.device)

        origins = rays_o[ray_indices]
        dirs = rays_d[ray_indices]
        if camera_indices is not None:
            camera_indices = camera_indices[ray_indices]

        zeros = torch.zeros_like(origins[:, :1])
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=dirs,
                starts=starts[..., None],
                ends=ends[..., None],
                pixel_area=zeros,
            ),
            camera_indices=camera_indices,
        )
        if ray_bundle.times is not None:
            ray_samples.times = ray_bundle.times[ray_indices]
        return ray_samples, ray_indices