from glob import glob
from pathlib import Path
from typing import Optional, Union, List, Dict

import numpy as np
import yaml
from elias.folder.model import ModelFolder, _ModelManagerType
from elias.manager.model import ModelManager, _ModelConfigType, _OptimizationConfigType, _ModelType
from elias.util import save_img, load_img, save_json, load_json

from nersemble.constants import EVALUATION_CAM_IDS
from nersemble.data_manager.multi_view_data import NeRSembleDataManager
from nersemble.env import NERSEMBLE_MODELS_PATH
from nersemble.model_manager.evaluation import NVSEvaluationResult
from nersemble.nerfstudio.config.nersemble_trainer_config import NeRSembleTrainerConfig


class NeRSembleBaseModelManager(ModelManager[None, None, None, None, None, None, NVSEvaluationResult]):

    def __init__(self, folder_name: str, run_name: str):
        models_path = NERSEMBLE_MODELS_PATH
        super().__init__(f"{models_path}/{folder_name}",
                         run_name,
                         checkpoint_name_format="step-$.ckpt",
                         checkpoints_sub_folder="checkpoints")
        self._folder_name = folder_name

    def _build_model(self,
                     model_config: _ModelConfigType,
                     optimization_config: Optional[_OptimizationConfigType] = None, **kwargs) -> _ModelType:
        raise NotImplementedError()

    def _store_checkpoint(self, model: _ModelType, checkpoint_file_name: str, **kwargs):
        raise NotImplementedError()

    def _load_checkpoint(self, checkpoint_file_name: Union[str, int], **kwargs) -> _ModelType:
        raise NotImplementedError()

    def load_config(self) -> 'NeRSembleTrainerConfig':
        with open(self.get_config_path()) as f:
            config = yaml.load(f, yaml.Loader)
        return config

    def save_config(self, config: 'NeRSembleTrainerConfig'):
        config_path = Path(self.get_config_path())
        config_path.write_text(yaml.dump(config), "utf8")

    def get_participant_id(self) -> int:
        config = self.load_config()
        return config.pipeline.datamanager.dataparser.participant_id

    def get_sequence_name(self) -> str:
        config = self.load_config()
        return config.pipeline.datamanager.dataparser.sequence_name

    def get_data_manager(self) -> NeRSembleDataManager:
        data_manager = NeRSembleDataManager(self.get_participant_id(), self.get_sequence_name())
        return data_manager

    def save_evaluation_img(self,
                            cam_id: int,
                            img: np.ndarray,
                            checkpoint: Union[str, int] = -1,
                            timestep: int = 0,
                            max_eval_timesteps: int = 15,
                            skip_timesteps: Optional[int] = None,
                            use_occupancy_grid_filtering: bool = True):
        evaluation_img_path = self.get_evaluation_img_path(cam_id, checkpoint, timestep,
                                                           max_eval_timesteps=max_eval_timesteps,
                                                           skip_timesteps=skip_timesteps,
                                                           use_occupancy_grid_filtering=use_occupancy_grid_filtering)
        save_img(img, evaluation_img_path)

    def get_evaluation_img_path(self,
                                cam_id: int,
                                checkpoint: Union[str, int] = -1,
                                timestep: int = 0,
                                max_eval_timesteps: int = 15,
                                skip_timesteps: Optional[int] = None,
                                use_occupancy_grid_filtering: bool = True) -> str:

        return f"{self.get_evaluation_folder(checkpoint, max_eval_timesteps=max_eval_timesteps, skip_timesteps=skip_timesteps, use_occupancy_grid_filtering=use_occupancy_grid_filtering)}/frame_{timestep:05d}/cam_{cam_id}.png"

    def load_evaluation_img(self, cam_id: int,
                            checkpoint:
                            Union[str, int] = -1,
                            timestep: int = 0,
                            use_alpha_map: bool = False,
                            use_alpha_channel: bool = False,
                            max_eval_timesteps: int = 15,
                            skip_timesteps: Optional[int] = None,
                            use_occupancy_grid_filtering: bool = True) -> np.ndarray:
        if checkpoint == -1:
            checkpoint = self.list_evaluated_checkpoint_ids()[-1]

        evaluation_img_path = self.get_evaluation_img_path(cam_id, checkpoint, timestep,
                                                           max_eval_timesteps=max_eval_timesteps,
                                                           skip_timesteps=skip_timesteps,
                                                           use_occupancy_grid_filtering=use_occupancy_grid_filtering)
        img = load_img(evaluation_img_path)

        if use_alpha_map:
            data_manager = self.get_data_manager()
            start_timestep = self.load_config().pipeline.datamanager.dataparser.start_timestep
            alpha_mask = data_manager.load_alpha_map(timestep + start_timestep, EVALUATION_CAM_IDS[cam_id])

            if use_alpha_channel:
                img = np.concatenate([img, alpha_mask[..., None]], axis=-1)
            else:
                alpha_mask = alpha_mask / 255.
                img = img / 255.
                img = alpha_mask[..., None] * img + (1 - alpha_mask[..., None])
                img = (img * 255).astype(np.uint8)

        return img

    def list_evaluated_timesteps(self, checkpoint: int = -1,
                                 max_eval_timesteps: int = 15,
                                 skip_timesteps: Optional[int] = None,
                                 use_occupancy_grid_filtering: bool = True) -> List[int]:
        timesteps = []
        folder_names = [path.name for path in
                        Path(self.get_evaluation_folder(checkpoint=checkpoint, max_eval_timesteps=max_eval_timesteps,
                                                        skip_timesteps=skip_timesteps,
                                                        use_occupancy_grid_filtering=use_occupancy_grid_filtering)).iterdir()
                        if
                        path.is_dir()]
        for folder_name in folder_names:
            if folder_name.startswith("frame_"):
                timestep = int(folder_name.split("_")[1])
                timesteps.append(timestep)

        timesteps = sorted(timesteps)
        return timesteps

    def load_evaluation_images(self,
                               cam_id: Optional[int] = None,
                               timestep: Optional[int] = None,
                               checkpoint: int = -1,
                               max_eval_timesteps: int = 15,
                               skip_timesteps: Optional[int] = None,
                               use_occupancy_grid_filtering: bool = True
                               ) -> List[np.ndarray]:
        if cam_id is None:
            cam_ids = [0, 1, 2, 3]
        else:
            cam_ids = [cam_id]

        if timestep is None:
            timesteps = self.list_evaluated_timesteps(checkpoint=checkpoint,
                                                      max_eval_timesteps=max_eval_timesteps,
                                                      skip_timesteps=skip_timesteps,
                                                      use_occupancy_grid_filtering=use_occupancy_grid_filtering)
        else:
            timesteps = [timestep]

        images = []
        for timestep in timesteps:
            for cam_id in cam_ids:
                img = self.load_evaluation_img(cam_id,
                                               timestep=timestep,
                                               checkpoint=checkpoint,
                                               max_eval_timesteps=max_eval_timesteps, skip_timesteps=skip_timesteps,
                                               use_occupancy_grid_filtering=use_occupancy_grid_filtering)
                images.append(img)

        return images

    def save_evaluation_result(self, evaluation_result: NVSEvaluationResult,
                               checkpoint: int = -1,
                               max_eval_timesteps: int = 15,
                               skip_timesteps: Optional[int] = None,
                               use_occupancy_grid_filtering: bool = True):
        save_json(evaluation_result.to_json(),
                  self.get_evaluation_result_path(
                      checkpoint,
                      max_eval_timesteps=max_eval_timesteps,
                      skip_timesteps=skip_timesteps,
                      use_occupancy_grid_filtering=use_occupancy_grid_filtering))

    def load_evaluation_result(self,
                               checkpoint: int = -1,
                               max_eval_timesteps: int = 15,
                               skip_timesteps: Optional[int] = None,
                               use_occupancy_grid_filtering: bool = True) -> NVSEvaluationResult:
        return NVSEvaluationResult.from_json(load_json(
            self.get_evaluation_result_path(checkpoint,
                                            max_eval_timesteps=max_eval_timesteps,
                                            skip_timesteps=skip_timesteps,
                                            use_occupancy_grid_filtering=use_occupancy_grid_filtering)))

    def list_evaluated_checkpoint_ids(self) -> List[int]:
        evaluations_path = Path(self.get_evaluations_folder())
        if not evaluations_path.exists():
            return []

        evaluated_checkpoint_ids = []
        for path in evaluations_path.iterdir():
            try:
                checkpoint_id = int(path.name.split('_')[1])
                evaluated_checkpoint_ids.append(checkpoint_id)
            except ValueError:
                pass

        return evaluated_checkpoint_ids

    def list_evaluations(self,
                         checkpoint: int = -1,
                         max_eval_timesteps: int = 15,
                         skip_timesteps: Optional[int] = None,
                         use_occupancy_grid_filtering: bool = True) -> List[str]:
        evaluation_file_paths = glob(
            f"{self.get_evaluation_folder(checkpoint, max_eval_timesteps=max_eval_timesteps, skip_timesteps=skip_timesteps, use_occupancy_grid_filtering=use_occupancy_grid_filtering)}/evaluation_result*.json")
        evaluations = []
        for evaluation_file_path in evaluation_file_paths:
            evaluation_file_name = Path(evaluation_file_path).stem
            evaluations.append(evaluation_file_name)

        return evaluations

    # ==========================================================
    # Paths
    # ==========================================================

    # TODO: continue
    def get_config_path(self) -> str:
        return f"{self._location}/config.yml"

    def get_path_to_rendering(self) -> str:
        candidate_files = glob(f"{FAMUDY_RENDERINGS_PATH}/{self._folder_name}/output_{self.get_run_name()}*.mp4")
        if len(candidate_files) > 1:
            print(f"Found multiple rendering outputs for {self.get_run_name()}")
            print(candidate_files)
            print("Returning the first")

        return candidate_files[0]

    def get_checkpoint_folder(self) -> str:
        checkpoint_folder = f"{self._location}/checkpoints"
        return checkpoint_folder

    def get_evaluations_folder(self) -> str:
        return f"{self._location}/evaluation"

    def get_evaluation_folder(self, checkpoint: Union[str, int] = -1,
                              max_eval_timesteps: int = 15,
                              skip_timesteps: Optional[int] = None,
                              use_occupancy_grid_filtering: bool = True) -> str:
        if checkpoint == -1:
            checkpoint = list(sorted(self.list_evaluated_checkpoint_ids()))[-1]

        name_parts = []

        if max_eval_timesteps > 0:
            name_parts.append(f"max_eval_timesteps_{max_eval_timesteps}")

        if skip_timesteps is not None and skip_timesteps > 1:
            name_parts.append(f"skip_timesteps_{skip_timesteps}")

        if not use_occupancy_grid_filtering:
            name_parts.append(f"no-occupancy-grid-filtering")

        if name_parts:
            name = '_'.join(name_parts)
        else:
            name = None

        checkpoint_folder_name = f"checkpoint_{checkpoint}"
        if name is not None:
            checkpoint_folder_name = f"{checkpoint_folder_name}_{name}"

        return f"{self.get_evaluations_folder()}/{checkpoint_folder_name}"

    def get_evaluation_result_path(self,
                                   checkpoint: int = -1,
                                   max_eval_timesteps: int = 15,
                                   skip_timesteps: Optional[int] = None,
                                   use_occupancy_grid_filtering: bool = True) -> str:

        return f"{self.get_evaluation_folder(checkpoint, max_eval_timesteps=max_eval_timesteps, skip_timesteps=skip_timesteps, use_occupancy_grid_filtering=use_occupancy_grid_filtering)}/evaluation_result.json"


class NeRSembleBaseModelFolder(ModelFolder[_ModelManagerType]):

    def __init__(self, model_folder: str, model_prefix: str):
        super().__init__(f"{NERSEMBLE_MODELS_PATH}/{model_folder}", model_prefix, True)

    def open_run(self, run_name_or_id: Union[str, int]) -> NeRSembleBaseModelManager:
        run_name = self.resolve_run_name(run_name_or_id)
        return self._cls_run_manager(run_name)

    def list_evaluated_run_ids(self) -> List[int]:
        run_ids = self.list_run_ids()
        evaluated_run_ids = []
        for run_id in run_ids:
            model_manager = self.open_run(run_id)
            evaluated_checkpoints = model_manager.list_evaluated_checkpoint_ids()
            if len(evaluated_checkpoints) > 0:
                evaluated_run_ids.append(run_id)

        return evaluated_run_ids
