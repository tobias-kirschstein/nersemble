from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nerfstudio.engine.trainer import TrainerConfig


@dataclass
class NeRSembleTrainerConfig(TrainerConfig):
    run_name: Optional[str] = None
    relative_model_dir: Path = Path("checkpoints/")

    def get_base_dir(self) -> Path:
        """Retrieve the base directory to set relative paths"""
        # check the experiment and method names
        assert self.run_name is not None, "Please set run_name in config"
        self.set_experiment_name()
        return Path(f"{self.output_dir}/{self.run_name}")

    def get_checkpoint_dir(self) -> Path:
        """Retrieve the checkpoint directory"""
        return Path(self.get_base_dir() / self.relative_model_dir)
