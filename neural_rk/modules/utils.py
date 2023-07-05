import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from neural_rk.dummy import DummyGradScaler
from neural_rk.hyperparameter import HyperParameter


def count_trainable_param(model: nn.Module) -> int:
    """Return number of trainable parameters of model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def equally_divide_df(df: pd.DataFrame, num_divide: int) -> list[int]:
    tot_num_nodes = sum(trajectory.shape[1] for trajectory in df.trajectories)
    nearly_equal_num_nodes = tot_num_nodes // num_divide

    divided_num_nodes = 0
    dividing_index = [0]
    for index, trajectory in enumerate(df.trajectories):
        divided_num_nodes += trajectory.shape[1]
        if divided_num_nodes > nearly_equal_num_nodes:
            dividing_index.append(index)
            divided_num_nodes = 0
    dividing_index.append(len(df))

    return dividing_index


def prune_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    consume_prefix_in_state_dict_if_present(state_dict, "approximator.")
    return state_dict


def load_optimizer(
    hp: HyperParameter, model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> optim.Optimizer:
    optimizer: optim.Optimizer = getattr(optim, hp.optimizer)(
        model.parameters(), lr=hp.scheduler.lr, weight_decay=hp.weight_decay
    )
    optimizer.load_state_dict(state_dict)
    return optimizer


def load_grad_scaler(
    use_amp: bool, device: torch.device, state_dict: dict[str, torch.Tensor]
) -> GradScaler | DummyGradScaler:
    grad_scaler = (
        GradScaler() if use_amp and device.type == "cuda" else DummyGradScaler()
    )
    grad_scaler.load_state_dict(state_dict)
    return grad_scaler
