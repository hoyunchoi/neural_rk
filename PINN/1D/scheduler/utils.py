import torch.nn as nn
import torch.optim as optim

from neural_rk.hyperparameter import SchedulerParameter

from .cosine import CosineScheduler
from .exponential import ExponentialScheduler
from .scheduler import Scheduler
from .step import StepScheduler


def get_scheduler(
    hp: SchedulerParameter, optimizer: optim.Optimizer | None = None
) -> Scheduler:
    if optimizer is None:
        optimizer = optim.SGD(nn.Linear(1, 1).parameters(), lr=hp.lr)

    if hp.name == "cosine":
        return CosineScheduler.from_hp(hp, optimizer)
    elif hp.name == "step":
        return StepScheduler.from_hp(hp, optimizer)
    else:
        return ExponentialScheduler.from_hp(hp, optimizer)
