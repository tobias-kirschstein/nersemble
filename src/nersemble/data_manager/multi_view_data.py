import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict

import numpy as np
from dreifus.matrix import Pose, Intrinsics
from elias.config import Config
from elias.util import load_json, load_img

from nersemble.constants import SERIALS
from nersemble.env import NERSEMBLE_DATA_PATH
from nersemble.util.quantization import DepthQuantizer

CAM_ID_SERIAL_TYPE = Union[str, int]


@dataclass
class CameraParams(Config):
    world_2_cam: Dict[str, Pose]
    intrinsics: Intrinsics


class NeRSembleDataManager:

    def __init__(self,
                 participant_id: int,
                 sequence_name: str):
        """
        The NeRSembleDataManager encapsulates the access to all data files.
        This includes multi-view video frames, depth maps, camera calibration, background masks etc.

        Expected folder layout:
        <NERSEMBLE_DATA_PATH>
         ├── 018
         │   ├── sequences
         │   │   └── EMO-1-shout+laugh
         │   │       ├── frame_00000
         │   │       │   ├── images-2x
         │   │       │   │   ├── cam_220700191.png              #
         │   │       │   │   :                                  # multi-view images
         │   │       │   │   └── cam_222200049.png              #
         │   │       │   ├── alpha_map
         │   │       │   │   ├── cam_220700191.png              #
         │   │       │   │   :                                  # alpha maps for background removal
         │   │       │   │   └── cam_222200049.png              #
         │   │       │   └── colmap
         │   │       │       └── depth_maps_compressed
         │   │       │           └── 12
         │   │       │               ├── cam_220700191.npz      #
         │   │       │               :                          # depth maps
         │   │       │               └── cam_222200049.npz      #
         │   │       ├── frame_00001
         │   │       │   :
         │   │       ├── frame_00002
         │   │       │   :
         │   │       :
         │   │
         │   ├── annotations
         │   │   └── EMO-1-shout+laugh
         │   │       └── color_correction
         │   │           ├── 220700191.npy                      #
         │   │           :                                      # color correction
         │   │           └── 220700191.npy                      #
         │   │
         │   └── camera_params.json                             # calibration
         │
         ├── 030
         │   :
         :
        """

        self._participant_id = participant_id
        self._sequence_name = sequence_name
        self._location = NERSEMBLE_DATA_PATH

    # ==========================================================
    # Assets
    # ==========================================================

    def load_image(self, timestep: int, cam_id_or_serial: CAM_ID_SERIAL_TYPE) -> np.ndarray:
        image_path = self.get_image_path(timestep, cam_id_or_serial)
        return load_img(image_path)

    def load_camera_params(self) -> CameraParams:
        camera_params = load_json(self.get_camera_params_path())
        intrinsics = Intrinsics(np.asarray(camera_params["intrinsics"]))
        world_2_cam = {serial: Pose(np.asarray(world_2_cam)) for serial, world_2_cam in
                       camera_params["world_2_cam"].items()}
        return CameraParams(world_2_cam, intrinsics)
        # return CameraParams.from_json(load_json(self.get_camera_params_path()),
        #                               type_hooks={
        #                                   Pose: lambda value: Pose(value),
        #                                   Intrinsics: lambda value: Intrinsics(value)})

    def load_alpha_map(self, timestep: int, cam_id_or_serial: CAM_ID_SERIAL_TYPE) -> np.ndarray:
        alpha_map_path = self.get_alpha_map_path(timestep, cam_id_or_serial)
        return load_img(alpha_map_path)

    def depth_map_exists(self, timestep: int, cam_id_or_serial: CAM_ID_SERIAL_TYPE) -> bool:
        depth_map_path = self.get_depth_map_path(timestep, cam_id_or_serial)
        return Path(depth_map_path).exists()

    def load_depth_map(self, timestep: int, cam_id_or_serial: CAM_ID_SERIAL_TYPE) -> np.ndarray:
        depth_map_path = self.get_depth_map_path(timestep, cam_id_or_serial)

        depth_quantizer = DepthQuantizer()
        return depth_quantizer.decode(load_img(depth_map_path))

        # return np.load(depth_map_path)['depth_map']

    # def load_consistency_graph(self, timestep: int, cam_id_or_serial: CAM_ID_SERIAL_TYPE) -> np.ndarray:
    #     consistency_graph_path = self.get_consistency_graph_path(timestep, cam_id_or_serial)
    #     return np.load(consistency_graph_path)['consistency_graph']

    # ==========================================================
    # Data structure
    # ==========================================================
    # Folders
    # ----------------------------------------------------------

    def get_participant_folder(self) -> str:
        return f"{self._location}/{self._participant_id:03d}"

    def get_sequence_folder(self) -> str:
        return f"{self.get_participant_folder()}/sequences/{self._sequence_name}"

    def get_timestep_folder(self, timestep: int) -> str:
        return f"{self.get_sequence_folder()}/frame_{timestep:05d}"

    def get_images_folder(self, timestep: int) -> str:
        return f"{self.get_timestep_folder(timestep)}/images-2x-73fps"

    def get_alpha_map_folder(self, timestep: int) -> str:
        return f"{self.get_timestep_folder(timestep)}/alpha_map-73fps"

    def get_colmap_folder(self, timestep: int) -> str:
        return f"{self.get_timestep_folder(timestep)}/colmap-73fps"

    def get_depth_maps_folder(self, timestep: int) -> str:
        # return f"{self.get_colmap_folder(timestep)}/depth_maps_geometric/{n_cameras}"
        return f"{self.get_colmap_folder(timestep)}/depth_maps_compressed"

    # def get_consistency_graphs_folder(self, timestep: int, n_cameras: int = 12) -> str:
    #     return f"{self.get_colmap_folder(timestep)}/consistency_graphs/{n_cameras}"

    def get_annotations_folder(self) -> str:
        return f"{self.get_participant_folder()}/annotations/{self._sequence_name}"

    def get_color_correction_folder(self) -> str:
        return f"{self.get_annotations_folder()}/color_correction"

    # ----------------------------------------------------------
    # Paths
    # ----------------------------------------------------------

    def get_image_path(self, timestep: int, cam_id_or_serial: CAM_ID_SERIAL_TYPE) -> str:
        serial = self.cam_id_to_serial(cam_id_or_serial)
        return f"{self.get_images_folder(timestep)}/cam_{serial}.png"

    def get_alpha_map_path(self, timestep: int, cam_id_or_serial: CAM_ID_SERIAL_TYPE) -> str:
        serial = self.cam_id_to_serial(cam_id_or_serial)
        return f"{self.get_alpha_map_folder(timestep)}/cam_{serial}.png"

    def get_depth_map_path(self, timestep: int, cam_id_or_serial: CAM_ID_SERIAL_TYPE) -> str:
        serial = self.cam_id_to_serial(cam_id_or_serial)
        # return f"{self.get_depth_maps_folder(timestep)}/cam_{serial}.npz"
        return f"{self.get_depth_maps_folder(timestep)}/cam_{serial}.png"
    #
    # def get_consistency_graph_path(self, timestep: int, cam_id_or_serial: CAM_ID_SERIAL_TYPE) -> str:
    #     serial = self.cam_id_to_serial(cam_id_or_serial)
    #     return f"{self.get_consistency_graphs_folder(timestep)}/cam_{serial}.npz"

    def get_color_correction_path(self, cam_id_or_serial: CAM_ID_SERIAL_TYPE) -> str:
        serial = self.cam_id_to_serial(cam_id_or_serial)
        return f"{self.get_color_correction_folder()}/{serial}.npy"

    def get_camera_params_path(self) -> str:
        return f"{self.get_participant_folder()}/camera_params.json"

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------

    def cam_id_to_serial(self, cam_id_or_serial: CAM_ID_SERIAL_TYPE) -> str:
        if isinstance(cam_id_or_serial, int):
            return SERIALS[cam_id_or_serial]
        else:
            return cam_id_or_serial

    def serial_to_cam_id(self, cam_id_or_serial: CAM_ID_SERIAL_TYPE) -> int:
        if isinstance(cam_id_or_serial, str):
            return SERIALS.index(cam_id_or_serial)
        else:
            return cam_id_or_serial

    def get_timesteps(self) -> List[int]:
        timestep_folder_regex = re.compile("frame_(\d+)")
        timesteps = []
        for timestep_folder in Path(self.get_sequence_folder()).iterdir():
            re_match = timestep_folder_regex.match(timestep_folder.name)
            if re_match:
                timestep = int(re_match.group(1))
                if Path(self.get_images_folder(timestep)).exists():
                    timesteps.append(timestep)

        timesteps = sorted(timesteps)
        return timesteps

    def get_n_timesteps(self) -> int:
        return len(self.get_timesteps())
