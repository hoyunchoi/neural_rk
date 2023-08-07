import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import bc_derivative_mse, pde_mse


def train_ic(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    field: torch.Tensor,
    num_data: int,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    idx = rng.choice(len(x), num_data, replace=False)
    x, t = x[idx].to(device), t[idx].to(device)
    field = field[idx].to(device)

    pred_field = model(x, t)

    loss_ic = F.mse_loss(pred_field, field)
    return loss_ic


def train_bc(
    model: nn.Module,
    x1: torch.Tensor,
    t1: torch.Tensor,
    x2: torch.Tensor,
    t2: torch.Tensor,
    num_data: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = rng.choice(len(x1), num_data, replace=False)
    x1 = x1[idx].to(device).requires_grad_(True)
    t1 = t1[idx].to(device).requires_grad_(True)
    x2 = x2[idx].to(device).requires_grad_(True)
    t2 = t2[idx].to(device).requires_grad_(True)

    pred_field1 = model(x1, t1)
    pred_field2 = model(x2, t2)

    loss_bc_value = F.mse_loss(pred_field1, pred_field2)
    loss_bc_derivative = bc_derivative_mse(x1, t1, pred_field1, x2, t2, pred_field2)
    return loss_bc_value, loss_bc_derivative


def train_pde_data(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    field: torch.Tensor,
    num_data: int,
    nu: torch.Tensor,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = rng.choice(len(x), num_data, replace=False)
    x = x[idx].to(device).requires_grad_(True)
    t = t[idx].to(device).requires_grad_(True)
    field = field[idx].to(device)

    pred_field = model(x, t)

    loss_pde = pde_mse(x, t, pred_field, nu)
    loss_data = F.mse_loss(pred_field, field)
    return loss_pde, loss_data


@torch.no_grad()
def validate(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    field: torch.Tensor,
    num_data: int,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    idx = rng.choice(len(x), num_data, replace=False)
    x, t = x[idx].to(device), t[idx].to(device)
    field = field[idx].to(device)

    pred_field = model(x, t)
    loss_val = F.mse_loss(pred_field, field)
    return loss_val
