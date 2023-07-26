import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from loss import bc_derivative_mse, pde_mse


def train_ic(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    losses: list[torch.Tensor] = []

    for x, y, t, field in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        field = field.to(device, non_blocking=True)
        pred_field = model(x, y, t)
        loss = F.mse_loss(pred_field, field)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())

    return torch.mean(torch.stack(losses)).item()


def train_bc(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer_value: optim.Optimizer,
    optimizer_derivative: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    losses_value: list[torch.Tensor] = []
    losses_derivative: list[torch.Tensor] = []
    for x1, y1, t1, x2, y2, t2 in dataloader:
        x1 = x1.requires_grad_(True).to(device, non_blocking=True)
        y1 = y1.requires_grad_(True).to(device, non_blocking=True)
        t1 = t1.requires_grad_(True).to(device, non_blocking=True)
        x2 = x2.requires_grad_(True).to(device, non_blocking=True)
        y2 = y2.requires_grad_(True).to(device, non_blocking=True)
        t2 = t2.requires_grad_(True).to(device, non_blocking=True)

        pred_field1 = model(x1, y1, t1)
        pred_field2 = model(x2, y2, t2)

        loss1 = F.mse_loss(pred_field1, pred_field2)
        loss2 = bc_derivative_mse(x1, y1, t1, pred_field1, x2, y2, t2, pred_field2)

        optimizer_value.zero_grad(set_to_none=True)
        optimizer_derivative.zero_grad(set_to_none=True)
        loss1.backward(retain_graph=True)
        loss2.backward()
        optimizer_value.step()
        optimizer_derivative.step()
        losses_value.append(loss1.detach())
        losses_derivative.append(loss2.detach())

    return (
        torch.mean(torch.stack(losses_value)).item(),
        torch.mean(torch.stack(losses_derivative)).item(),
    )


def train_pde(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer_pde: optim.Optimizer,
    optimizer_data: optim.Optimizer,
    nu: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    losses_pde: list[torch.Tensor] = []
    losses_data: list[torch.Tensor] = []
    for x, y, t, field in dataloader:
        x = x.requires_grad_(True).to(device, non_blocking=True)
        y = y.requires_grad_(True).to(device, non_blocking=True)
        t = t.requires_grad_(True).to(device, non_blocking=True)
        field = field.to(device, non_blocking=True)
        pred_field = model(x, y, t)

        loss1 = pde_mse(x, y, t, pred_field, nu)
        loss2 = F.mse_loss(pred_field, field)

        optimizer_pde.zero_grad(set_to_none=True)
        optimizer_data.zero_grad(set_to_none=True)
        loss1.backward(retain_graph=True)
        loss2.backward()
        optimizer_pde.step()
        optimizer_data.step()
        losses_pde.append(loss1.detach())
        losses_data.append(loss2.detach())

    return (
        torch.mean(torch.stack(losses_pde)).item(),
        torch.mean(torch.stack(losses_data)).item(),
    )


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    losses: list[torch.Tensor] = []
    for x, y, t, field in dataloader:
        x, y, t, field = x.to(device), y.to(device), t.to(device), field.to(device)
        pred_field = model(x, y, t)

        loss = F.mse_loss(pred_field, field)
        losses.append(loss)
    return torch.mean(torch.stack(losses)).item()
