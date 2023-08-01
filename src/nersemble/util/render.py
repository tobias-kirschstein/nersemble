from typing import List, Optional

import mediapy
import numpy as np
import torch
from elias.util import ensure_directory_exists_for_file
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.base_model import Model
from nerfstudio.utils.colormaps import apply_depth_colormap, ColormapOptions
from tqdm import tqdm


def render_trajectory_video(model: Model,
                            cameras: Cameras,
                            output_path: str,
                            rendered_resolution_scaling_factor: float = 1.0,
                            render_channels: List[str] = None,
                            seconds: Optional[float] = None):
    print(f"Storing rendered videos in: {output_path}")

    if render_channels is None:
        render_channels = ['rgb']

    if seconds is None:
        fps = 24
    else:
        fps = len(cameras) / seconds

    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(model.device)

    ensure_directory_exists_for_file(output_path)
    writers = dict()

    for camera_idx in tqdm(range(cameras.size), desc="Rendering video"):
        camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)

        with torch.no_grad():
            outputs = model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        for render_channel in render_channels:
            assert render_channel in outputs, f"Could not find {render_channel} in the model outputs"

            output_image = outputs[render_channel].cpu().numpy()
            if output_image.shape[-1] == 1:
                output_image = np.concatenate((output_image,) * 3, axis=-1)

            if render_channel == 'depth':
                output_image = apply_depth_colormap(
                    outputs[render_channel],
                    accumulation=outputs["accumulation"],
                    near_plane=0.8*9,  # TODO: Would be better to pass scale factor to function
                    far_plane=1.2*9,
                    colormap_options=ColormapOptions(colormap="turbo", invert=True)
                ).cpu().numpy()

            if render_channel not in writers:
                render_width = int(output_image.shape[1])
                render_height = int(output_image.shape[0])
                writer = mediapy.VideoWriter(
                    path=output_path.format(r=render_channel),
                    shape=(render_height, render_width),
                    fps=fps,
                )
                writer.__enter__()
                writers[render_channel] = writer
            else:
                writer = writers[render_channel]

            writer.add_image(output_image)

    for writer in writers.values():
        writer.__exit__()