import pathlib
from pathlib import Path
from typing import Optional, Literal, Tuple

import torch
import yaml
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_load_checkpoint

from nersemble.nerfstudio.config.nersemble_trainer_config import NeRSembleTrainerConfig


def nersemble_eval_setup(
        config_path: Path,
        checkpoint_folder: str,
        eval_num_rays_per_chunk: Optional[int] = None,
        test_mode: Literal["test", "val", "inference"] = "test",
        checkpoint: Optional[int] = None,
        scene_box: torch.Tensor = None,
        max_eval_timesteps: Optional[int] = None,
        eval_num_images_to_sample_from: Optional[int] = None,
) -> Tuple[NeRSembleTrainerConfig, Pipeline, Path, int]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    config = try_load_config(config_path)

    if max_eval_timesteps is not None:
        config.pipeline.datamanager.dataparser.max_eval_timesteps = max_eval_timesteps

    if eval_num_images_to_sample_from is not None:
        config.pipeline.datamanager.eval_num_images_to_sample_from = eval_num_images_to_sample_from

    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    # Nerfstudio stores the absolute path to the model checkpoint directory
    # This is wrong if the model files are moved somewhere else and NERSEMBLE_MODELS_PATH is updated accordingly
    # Hence, the path to the checkpoints has to be specified explicitly when loading models
    config.load_dir = Path(checkpoint_folder)

    if isinstance(config.pipeline.datamanager, VanillaDataManagerConfig):
        config.pipeline.datamanager.eval_image_indices = None

    if scene_box is not None:
        config.pipeline.datamanager.dataparser.scene_box = scene_box

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    config.load_step = checkpoint
    checkpoint_path, step = eval_load_checkpoint(config, pipeline)

    return config, pipeline, checkpoint_path, step


def try_load_config(config_path: str) -> NeRSembleTrainerConfig:
    config_text = pathlib.Path(config_path).read_text()
    posix_path = pathlib.PosixPath
    try:
        config = yaml.load(config_text, Loader=yaml.Loader)
    except NotImplementedError:
        # nerfstudio persists Path objects which is dumb since they are not platform independent
        # This is a hack to make yaml.Loader believe it can just construct the serialized PosixPath
        # into a WindowsPath (in case the config was created on a Linux machine and is loaded in Windows)
        pathlib.PosixPath = pathlib.WindowsPath
        config = yaml.load(config_text, Loader=yaml.Loader)
    finally:
        pathlib.PosixPath = posix_path

    return config
