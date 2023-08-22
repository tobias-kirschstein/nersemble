from pathlib import Path

import numpy as np
import torch
import tyro
from dreifus.camera import CameraCoordinateConvention
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3
from nerfacc import OccGridEstimator
from nerfstudio.cameras.cameras import Cameras

from nersemble.data_manager.multi_view_data import NeRSembleDataManager
from nersemble.env import NERSEMBLE_RENDERS_PATH
from nersemble.model_manager.nersemble import NeRSembleModelFolder
from nersemble.util.connected_components import filter_occupancy_grid
from nersemble.util.render import render_trajectory_video
from nersemble.util.setup import nersemble_eval_setup


def main(run_name: str,
         /,
         seconds: int = 4,
         fps: int = 24,
         n_rays: int = 2 ** 13,
         downscale_factor: int = 4,
         render_depth: bool = False,
         render_deformations: bool = False,
         use_occupancy_grid_filtering: bool = False,
         occupancy_grid_filtering_threshold: float = 0.05,
         occupancy_grid_filtering_sigma_erosion: int = 7,
         ):
    model_manager = NeRSembleModelFolder().open_run(run_name)

    # Setup pipeline and load model
    config_path = Path(model_manager.get_config_path())

    _, pipeline, _, checkpoint = nersemble_eval_setup(config_path,
                                                      model_manager.get_checkpoint_folder(),
                                                      eval_num_rays_per_chunk=n_rays,
                                                      eval_num_images_to_sample_from=1)

    dataparser_config = pipeline.datamanager.dataparser.config

    if use_occupancy_grid_filtering:
        # Ensure that eval occupancy grid only contains one large blob and no floaters
        occupancy_grid: OccGridEstimator = pipeline.model.occupancy_grid
        filter_occupancy_grid(occupancy_grid,
                              threshold=occupancy_grid_filtering_threshold,
                              sigma_erosion=occupancy_grid_filtering_sigma_erosion)

    render_output_folder = NERSEMBLE_RENDERS_PATH

    additional_label = ""

    if use_occupancy_grid_filtering:
        additional_label = f"{additional_label}_occ_grid_filtering"
    if checkpoint is not None:
        additional_label = f"{additional_label}_checkpoint-{checkpoint}"

    output_path = f"{render_output_folder}/{run_name}_{{r:}}{additional_label}.mp4"

    # Build trajectory
    n_timesteps = int(seconds * fps)
    cam_2_world_poses = circle_around_axis(n_timesteps,
                                           axis=Vec3(0, 1, 0),
                                           up=Vec3(0, 0, 1),
                                           move=Vec3(0, -1, 0),
                                           distance=0.3)
    cam_2_world_poses = [pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL, inplace=False) for
                         pose in cam_2_world_poses]
    cam_2_world_poses = np.stack(cam_2_world_poses)
    cam_2_world_poses[:, :3, 3] *= dataparser_config.scale_factor

    data_manager = NeRSembleDataManager(dataparser_config.participant_id, dataparser_config.sequence_name)
    intrinsics = data_manager.load_camera_params().intrinsics

    times = torch.arange(n_timesteps) / (n_timesteps - 1)

    cameras = Cameras(torch.tensor(cam_2_world_poses[:, :3, :4]),
                      fx=intrinsics.fx, fy=intrinsics.fy, cx=intrinsics.cx, cy=intrinsics.cy,
                      width=2200, height=3208,
                      times=times)

    render_channels = ["rgb"]
    if render_depth:
        render_channels.append("depth")
    if render_deformations:
        render_channels.append("deformation")

    render_trajectory_video(pipeline.model,
                            cameras,
                            output_path,
                            rendered_resolution_scaling_factor=1.0 / downscale_factor,
                            render_channels=render_channels,
                            seconds=seconds)


if __name__ == '__main__':
    tyro.cli(main)
