from dataclasses import dataclass, field
from typing import Type

from nerfstudio.engine.schedulers import SchedulerConfig, Scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR


@dataclass
class StepLRSchedulerConfig(SchedulerConfig):
    step_size: int = 10000
    _target: Type = field(default_factory=lambda: StepLRScheduler)
    gamma: float = 1e-1
    last_epoch: int = -1


class StepLRScheduler(Scheduler):
    config: StepLRSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        return StepLR(optimizer, self.config.step_size, gamma=self.config.gamma, last_epoch=self.config.last_epoch)
