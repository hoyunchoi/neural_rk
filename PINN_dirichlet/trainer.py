import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import pde_mse


def train_ic(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    field: torch.Tensor,
    num_data: int,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    idx = rng.choice(len(x), num_data, replace=False)
    x, y, t = x[idx].to(device), y[idx].to(device), t[idx].to(device)
    field = field[idx].to(device)

    pred_field = model(x, y, t)

    loss_ic = F.mse_loss(pred_field, field)
    return loss_ic


def train_bc(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    num_data: int,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    idx = rng.choice(len(x), num_data, replace=False)
    x = x[idx].to(device)
    y = y[idx].to(device)
    t = t[idx].to(device)

    pred_field = model(x, y, t)
    return (pred_field).square().mean()


def train_pde_data(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    field: torch.Tensor,
    num_data: int,
    nu: torch.Tensor,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = rng.choice(len(x), num_data, replace=False)
    x = x[idx].to(device).requires_grad_(True)
    y = y[idx].to(device).requires_grad_(True)
    t = t[idx].to(device).requires_grad_(True)
    field = field[idx].to(device)

    pred_field = model(x, y, t)

    loss_pde = pde_mse(x, y, t, pred_field, nu)
    loss_data = F.mse_loss(pred_field, field)
    return loss_pde, loss_data


@torch.no_grad()
def validate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    field: torch.Tensor,
    num_data: int,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    idx = rng.choice(len(x), num_data, replace=False)
    x, y, t = x[idx].to(device), y[idx].to(device), t[idx].to(device)
    field = field[idx].to(device)

    pred_field = model(x, y, t)
    loss_val = F.mse_loss(pred_field, field)
    return loss_val
