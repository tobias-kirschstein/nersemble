from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Optional, List, Type, Literal, Dict

import numpy as np
import torch
from PIL import Image
from dreifus.camera import CameraCoordinateConvention
from dreifus.graphics import Dimensions
from elias.config import implicit
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CameraType, Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, DataParser, DataParserConfig
from nerfstudio.data.scene_box import SceneBox

from nersemble.constants import EVALUATION_CAM_IDS, COMPLETE_CAM_ID_ORDER, SERIALS
from nersemble.data_manager.multi_view_data import NeRSembleDataManager
from nersemble.nerfstudio.model_components.frustum import TorchFrustum


@dataclass
class NeRSembleDataParserOutputs(DataparserOutputs):
    alpha_channel_filenames: Optional[List[Path]] = None
    """image_filenames can have RGBA images. 
        In case the alpha channels are stored separately, the corresponding paths can be specified here"""
    color_correction_filenames: Optional[List[Path]] = None
    """image_filenames can have associated color corrections (affine transformation matrices). 
        The corresponding paths can be specified here"""
    camera_frustums: Optional[List[TorchFrustum]] = None

@dataclass
class NeRSembleDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: NeRSembleDataParser)
    """target class to instantiate"""

    data: Optional[Path] = None  # Overwrite nerstudio's data field. We don't need it

    participant_id: int = -1  # ID of the participant
    sequence_name: str = "<<<SPECIFY_SEQUENCE_NAME>>>"  # Name of the sequence
    n_timesteps: int = 1  #
    n_cameras: int = 12  # How many cameras should be used for training
    skip_timesteps: int = 1  # If >1, only every n-th image will be used (i.e., the dataset is temporally downsampled)
    start_timestep: int = 0  # If other than 0, the first n timesteps will be skipped
    max_eval_timesteps: int = 3  # Eval dataset will only contain images from at most that many evenly spaced timesteps
    downscale_factor: int = 2
    scale_factor: int = 1  # By how much the world space should be scaled

    # Whether to only render points that can be seen from at least 1 train view (during inference)
    use_view_frustum_culling: bool = True
    scene_box: Optional[torch.Tensor] = None

    # Whether to load the background images for foreground/background segmentation
    foreground_only: bool = True
    use_depth_maps: bool = implicit(False)  # Whether to load depth maps (Should be set to True if depth loss is active)
    # depth_min_consistency: int = 3  # Only use computed depth values for pixels that are seen by at least that many cameras
    use_color_correction: bool = True
    use_alpha_maps: bool = implicit(False)  # Whether to load alpha maps (Should be set to True if alpha loss is active)
    alpha_channel_color: Literal['black', 'white'] = 'white'  # Images for NeRF will be alpha blended with that color
    alpha_map_threshold: int = 128  # If mask_mode == 'robust_matting' This defines the cutoff when pixels will be interpreted as foreground

    # ray_importance_sampling_type: Literal['std', 'mean', 'median', None] = None
    # geman_mclure_gamma: float = 1e-1
    # use_pixel_sample_probabilities: bool = False  # deprecated

    def image_idx_to_cam_id(self, image_idx: int, split: str = 'train') -> int:
        n_cameras = self.n_cameras if split == 'train' else len(EVALUATION_CAM_IDS)
        assert self.n_timesteps == -1 or (image_idx < n_cameras * self.n_timesteps), \
            f"Got larger image_idx ({image_idx}) than expected " \
            f"for n_cameras: {n_cameras} and n_timesteps: {self.n_timesteps} in split {split}"

        # Multiple frames of the same original camera will actually get the same cam_id
        cam_id = image_idx % n_cameras

        return cam_id

    def image_idx_to_timestep(self,
                              image_idx: int,
                              split: str = 'train') -> int:
        """
        Map image_idx (i.e., the ascending sample ID that identifies every single train sample) to the timestep that
        this image was taken at. This is relevant for mult-image datasets where multiple cameras take pictures at the
        exact timestep. Hence, there is a many-to-one mapping from image_idx to timestep
        """

        n_cameras = self.n_cameras if split == 'train' else len(EVALUATION_CAM_IDS)
        assert self.n_timesteps == -1 or (image_idx < n_cameras * self.n_timesteps), \
            f"Got larger image_idx ({image_idx}) than expected " \
            f"for n_cameras: {n_cameras} and n_timesteps: {self.n_timesteps} in split {split}"

        timestep = int(image_idx / n_cameras)

        if not split == 'train' and 0 < self.max_eval_timesteps < self.n_timesteps:
            # If not all images are used for evaluation, need to map the eval image_idx to the correct train timestep
            # Example:
            #   train timesteps: 0 1 2 3 4 5 6
            #   eval timesteps:  0 - - 1 - - 2
            #       eval_timestep(0) -> train_timstep(0)
            #       eval_timestep(1) -> train_timstep(3)
            #       eval_timestep(2) -> train_timstep(6)
            assert self.n_timesteps != -1, "Unknown number of timesteps"
            idx_timesteps_eval = np.linspace(0, self.n_timesteps - 1, self.max_eval_timesteps, dtype=int)
            timestep = idx_timesteps_eval[timestep]

        return timestep

    def get_timestep_to_original_mapping(self, n_effective_timesteps: int, split: str = 'train') -> List[int]:
        """
        Take into account start timestep to figure out the "original" timestep that maps to the actual image file.
        """

        timesteps = list(range(self.start_timestep,
                               (n_effective_timesteps + self.start_timestep) * self.skip_timesteps,
                               self.skip_timesteps))

        if not split == 'train' and 0 < self.max_eval_timesteps < len(timesteps):
            idx_timesteps_eval = np.linspace(0, len(timesteps) - 1, self.max_eval_timesteps, dtype=int)
            timesteps = [timesteps[idx] for idx in idx_timesteps_eval]

        return timesteps

    def original_timestep_to_time(self, timestep: int, split: str = 'train') -> float:
        original_timesteps = self.get_timestep_to_original_mapping(self.n_timesteps, split=split)
        min_timestep = min(original_timesteps)
        max_timestep = max(original_timesteps)
        time = (timestep - min_timestep) / (max_timestep - min_timestep) if timestep > min_timestep else 0

        return time

    def time_to_original_timestep(self, time: float, split: str ='train') -> int:
        original_timesteps = self.get_timestep_to_original_mapping(self.n_timesteps, split=split)
        min_timestep = min(original_timesteps)
        max_timestep = max(original_timesteps)

        timestep = round(time * (max_timestep - min_timestep)) + min_timestep
        return timestep

class NeRSembleDataParser(DataParser):
    config: NeRSembleDataParserConfig
    includes_time: bool = True  # Tells the viewer that the "times" field for ray_bundles can be used -> dynamics

    def __init__(self, config: NeRSembleDataParserConfig):
        super().__init__(config)

        self._data_manager = NeRSembleDataManager(config.participant_id, config.sequence_name)

        if config.n_timesteps == -1:
            n_total_timesteps = self._data_manager.get_n_timesteps()
            n_effective_timesteps = ceil(n_total_timesteps / config.skip_timesteps)
            config.n_timesteps = n_effective_timesteps
        else:
            n_effective_timesteps = config.n_timesteps
        self._n_effective_timesteps = n_effective_timesteps

        self._original_image_size = Dimensions(2200, 3208)
        self._image_size = Dimensions(self._original_image_size.w // self.config.downscale_factor,
                                      self._original_image_size.h // self.config.downscale_factor)

    def _generate_dataparser_outputs(self, split="train") -> NeRSembleDataParserOutputs:
        original_timesteps = self.config.get_timestep_to_original_mapping(self._n_effective_timesteps, split=split)

        # -----------------
        # Image file names
        # -----------------
        if split == 'train':
            cam_ids = COMPLETE_CAM_ID_ORDER[:self.config.n_cameras]
        else:
            cam_ids = EVALUATION_CAM_IDS

        image_filenames = []
        alpha_channel_filenames = []
        color_correction_filenames = []
        for timestep in original_timesteps:
            for cam_id in cam_ids:
                image_filenames.append(self._data_manager.get_image_path(timestep, cam_id))

                if self.config.foreground_only:
                    alpha_channel_filenames.append(self._data_manager.get_alpha_map_path(timestep, cam_id)
                                                   )
                if self.config.use_color_correction:
                    color_correction_filenames.append(self._data_manager.get_color_correction_path(cam_id))

        # --------
        # Cameras
        # --------

        camera_params = self._data_manager.load_camera_params()

        # -----------
        # Extrinsics
        # -----------
        # Calibration poses are OpenCV: x -> right, y -> down, z -> forward
        # nerfstudio uses OpenGL convention: x -> right, y -> UP, z -> BACK
        # nerfstudio viewer uses: x -> right, y -> FORWARD -> z -> UP
        # Here, we map the calibration poses to nerfstudio viewer coordinate convention

        world_to_cam_poses = [camera_params.world_2_cam[SERIALS[cam_id]] for cam_id in cam_ids]
        cam_to_world_poses = [world_to_cam.invert() for world_to_cam in world_to_cam_poses]

        for i in range(len(cam_to_world_poses)):
            cam_to_world_pose = cam_to_world_poses[i]
            # Negating the orientation axes is necessary due to different camera convention.
            # This only affects the rotation matrix and does not move the cameras
            cam_to_world_pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL)
            # cam_to_world_pose.negate_orientation_axis(1)  # y: down -> up
            # cam_to_world_pose.negate_orientation_axis(2)  # z: forward -> backward

            # Now "move" the cameras by swapping/inverting axes
            # This is only relevant for viewing the 3D reconstruction in nerfstudio's viewer
            cam_to_world_pose.swap_axes(['x', '-z', 'y'])  # y: up -> forward, z: backward -> up

            # Scale pose
            cam_to_world_pose[:3, 3] *= self.config.scale_factor

        # -----------
        # Intrinsics
        # -----------

        # All cameras have the same intrinsics
        intrinsic_params = [camera_params.intrinsics for _ in cam_ids]
        fxs = []
        fys = []
        cxs = []
        cys = []
        for intrinsic_param in intrinsic_params:
            fxs.append(intrinsic_param.fx)
            fys.append(intrinsic_param.fy)
            cxs.append(intrinsic_param.cx)
            cys.append(intrinsic_param.cy)

        # Intrinsics are given wrt the full resolution
        camera_type = CameraType.PERSPECTIVE

        # -----------
        # Distortion
        # -----------
        distortion_params = camera_utils.get_distortion_params(
            k1=0.0,
            k2=0.0,
            k3=0.0,
            k4=0.0,
            p1=0.0,
            p2=0.0,
        )

        # ----------------------------------------------------------
        # Camera Frustums
        # ----------------------------------------------------------
        camera_frustums = None
        if self.config.use_view_frustum_culling:
            camera_frustums = []
            for cam_pose, intrinsic_param in zip(cam_to_world_poses, intrinsic_params):
                cam_pose = cam_pose.copy().change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV)
                camera_frustums.append(TorchFrustum(torch.tensor(cam_pose),
                                                    torch.tensor(intrinsic_param),
                                                    self._original_image_size))

        fxs = torch.tensor(fxs)
        fys = torch.tensor(fys)
        cxs = torch.tensor(cxs)
        cys = torch.tensor(cys)
        cam_to_world_poses = torch.from_numpy(np.array(cam_to_world_poses).astype(np.float32))
        camera_to_worlds = cam_to_world_poses[:, :3, :4]

        n_effective_timesteps = len(original_timesteps)
        min_timestep = min(original_timesteps)
        max_timestep = max(original_timesteps)

        # Scale timesteps [min, max] to [0, 1]
        times = [
            (timestep - min_timestep) / (max_timestep - min_timestep) if timestep > min_timestep else 0
            for timestep in original_timesteps]

        # When using nerfstudio times, each frame from each viewpoint is modeled as a separate "Camera"
        # (even though the camera is still the same across views and its position did not change)
        # Hence, we have to repeat all the camera parameters for n_timesteps
        fxs = fxs.repeat(n_effective_timesteps)
        fys = fys.repeat(n_effective_timesteps)
        cxs = cxs.repeat(n_effective_timesteps)
        cys = cys.repeat(n_effective_timesteps)
        camera_to_worlds = camera_to_worlds.repeat(n_effective_timesteps, 1, 1)

        # "Cameras" from the same timestep will get the same "time"
        times = torch.tensor(times).repeat_interleave(len(cam_ids))

        cameras = Cameras(
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            distortion_params=distortion_params,
            height=self._original_image_size.h,
            width=self._original_image_size.w,
            camera_to_worlds=camera_to_worlds,
            camera_type=camera_type,
            times=times
        )
        cameras.rescale_output_resolution(1. / self.config.downscale_factor)

        # ----------
        # Scene box
        # ----------
        # in x,y,z order
        # assumes that the scene is centered at the origin
        if self.config.scene_box is not None:
            scene_box = SceneBox(self.config.scene_box)
        else:
            aabb_scale = self.config.scene_scale
            scene_box = SceneBox(
                aabb=torch.tensor(
                    [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]],
                    dtype=torch.float32
                )
            )

        outputs = NeRSembleDataParserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            camera_frustums=camera_frustums
        )

        if self.config.foreground_only:
            outputs.alpha_channel_filenames = alpha_channel_filenames
            if self.config.alpha_channel_color == 'black':
                outputs.alpha_color = torch.tensor([0.0, 0.0, 0.0])
            elif self.config.alpha_channel_color == 'white':
                outputs.alpha_color = torch.tensor([1.0, 1.0, 1.0])
            else:
                raise ValueError("Unknown alpha_channel_color: ", self.config.alpha_channel_color)

        if self.config.use_color_correction:
            outputs.color_correction_filenames = color_correction_filenames

        # --------------------------------
        # Additional data for supervision
        # --------------------------------

        # Depth Maps
        if self.config.use_depth_maps:
            outputs.metadata["depth_maps"] = {
                "func": self._load_depth_map,
                "kwargs": dict(split=split)
            }

        # Alpha Maps
        if self.config.use_alpha_maps:
            outputs.metadata["alpha_maps"] = {
                "func": self._load_mask,
                "kwargs": dict(split=split)
            }

        outputs.metadata["cam_ids"] = {
            "func": lambda image_idx, split: {"cam_ids": self.config.image_idx_to_cam_id(image_idx, split=split)},
            "kwargs": dict(split=split)
        }

        # NOTE: Don't forget to provide split=split to your newly created data loading lambda!

        return outputs

    # ==========================================================
    # Asset loaders
    # ==========================================================

    def _load_mask(self, image_idx: int, split: str = 'train') -> Dict[str, np.ndarray]:
        ret_dict = {}

        i_cam = self.config.image_idx_to_cam_id(image_idx, split=split)
        timestep = self.config.image_idx_to_timestep(image_idx, split=split)
        original_timestep = self.config.get_timestep_to_original_mapping(self._n_effective_timesteps)[timestep]

        if split == 'train':
            cam_id = COMPLETE_CAM_ID_ORDER[i_cam]
        else:
            cam_id = EVALUATION_CAM_IDS[i_cam]

        alpha_map = self._data_manager.load_alpha_map(original_timestep, cam_id)

        alpha_map = Image.fromarray(alpha_map)
        alpha_map = alpha_map.resize(self._image_size, resample=Image.BILINEAR)
        alpha_map = np.asarray(alpha_map)

        alpha_map = np.expand_dims(alpha_map, 2)

        return {
            "alpha_map": alpha_map,
        }

    def _load_depth_map(self, image_idx: int, split: str = 'train') -> Dict[str, np.ndarray]:
        # image_idx is just sequential index, need to map to actual cam_id

        if not split == 'train':
            # We only have depth maps for train views
            return dict()

        i_cam = self.config.image_idx_to_cam_id(image_idx, split=split)
        timestep = self.config.image_idx_to_timestep(image_idx, split=split)
        original_timestep = self.config.get_timestep_to_original_mapping(self._n_effective_timesteps)[timestep]

        cam_id = COMPLETE_CAM_ID_ORDER[i_cam]
        serial = SERIALS[cam_id]

        if not self._data_manager.depth_map_exists(original_timestep, serial):
            print(f"[WARNING] - No depth map found for timestep {original_timestep} and camera {serial}")
            depth_map = np.zeros((self._image_size.h, self._image_size.w))
        else:
            depth_map = self._data_manager.load_depth_map(original_timestep, serial)

            # Consistency filtering
            # consistency_graph = self._data_manager.load_consistency_graph(original_timestep, serial)
            # depth_map[consistency_graph < self.config.depth_min_consistency] = 0

            image = Image.fromarray(depth_map)
            image = image.resize(self._image_size, resample=Image.NEAREST)
            depth_map = np.array(image)

            outlier_mask = (depth_map < 0.8) | (1.4 < depth_map)  # Points further away than 1.4m should be ignored
            depth_map[outlier_mask] = 0

        # Scale depth values by how far cameras are scaled out
        depth_map *= self.config.scale_factor

        return {
            "depth_maps": depth_map
        }
