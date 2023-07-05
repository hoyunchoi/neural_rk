import functools
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch_scatter import scatter_sum

from neural_rk.grid_graph import grid2node, node2grid

arr = npt.NDArray[np.float32]


def clean(x: torch.Tensor) -> torch.Tensor:
    """Inf, NaN to zero"""
    return x.nan_to_num(0.0, posinf=0.0, neginf=0.0)


def burgers_graph(
    edge_list: torch.Tensor,
    edge_attr: torch.Tensor,
    nu: torch.Tensor,
    field: torch.Tensor,
) -> torch.Tensor:
    """
    edge_list: [E, 2], directed
    edge_attr: [E, 2], distance of the edge. 0 if not connected
    nu: [2, ] diffusion coefficient
    field: [N, 2]

    Return
    delta_field: [N, 2]
    """
    row, col = edge_list.T  # [E, ], [E, ]
    nu = nu.unsqueeze(0).unsqueeze(0)  # [2, ] -> [1, 1, 2]
    edge_attr = edge_attr.unsqueeze(1)   # [E, 2] -> [E, 1, 2]

    inv_spacing = clean(1.0 / edge_attr)  # [E, 1, 2]
    inv_double_spacing = clean(  # [N, 1, 2]
        1.0 / (scatter_sum(edge_attr, row, dim=0) + scatter_sum(edge_attr, col, dim=0))
    )

    # Half derivatives: [E, 1, 2] -> [N, 2, 2,], [N, 2, 2]
    derivative_per_edge = inv_spacing * (field[col] - field[row]).unsqueeze(-1)
    half_jacobian_minus = scatter_sum(derivative_per_edge, col, dim=0)
    half_jacobian_plus = scatter_sum(derivative_per_edge, row, dim=0)

    # (Double) derivatives: [E, 1, 2], [E, 1, 2] -> [N, 1, 2]
    jacobian = inv_double_spacing * (
        scatter_sum(edge_attr * half_jacobian_plus[col], col, dim=0)
        + scatter_sum(edge_attr * half_jacobian_minus[row], row, dim=0)
    )
    hessians = 2.0 * inv_double_spacing * (half_jacobian_plus - half_jacobian_minus)

    # Return: [N, 1, 2] -> [N, 2]
    return torch.sum(-field.unsqueeze(1) * jacobian + nu * hessians, dim=2)


def rk1(
    edge_list: torch.Tensor,
    edge_attr: torch.Tensor,
    nu: torch.Tensor,
    field: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    edge_list: [E, 2], directed
    edge_attr: [E, 2]
    nu: [2, ] diffusion coefficient
    field: [N, 2]
    dt

    Return: [N, 2] next field
    """
    delta_field = burgers_graph(edge_list, edge_attr, nu, field)
    return field + dt * delta_field


def rk2(
    edge_list: torch.Tensor,
    edge_attr: torch.Tensor,
    nu: torch.Tensor,
    field: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    edge_list: [E, 2], directed
    edge_attr: [E, 2]
    nu: [2, ] diffusion coefficient
    field: [N, 2]
    dt

    Return: [N, 2] next field
    """
    delta_field1 = burgers_graph(edge_list, edge_attr, nu, field)

    temp_field = field + dt * delta_field1
    delta_field2 = burgers_graph(edge_list, edge_attr, nu, temp_field)

    delta_field = 0.5 * (delta_field1 + delta_field2)
    return field + dt * delta_field


def rk4(
    edge_list: torch.Tensor,
    edge_attr: torch.Tensor,
    nu: torch.Tensor,
    field: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    edge_list: [E, 2], directed
    edge_attr: [E, 2] = [2*Nx*Ny, 2]\\
        [i, 0]: distance of i-th edge. Nonzero if i-th edge is x-axis\\
        [i, 1]: distance of i-th edge. Nonzero if i-th edge is y-axis\\
    edge_attr: [E, 2]
    nu: [2, ] diffusion coefficient
    field: [N, 2]
    dt

    Return: [N, 2] next field
    """
    delta_field1 = burgers_graph(edge_list, edge_attr, nu, field)

    temp_field = field + 0.5 * dt * delta_field1
    delta_field2 = burgers_graph(edge_list, edge_attr, nu, temp_field)

    temp_field = field + 0.5 * dt * delta_field2
    delta_field3 = burgers_graph(edge_list, edge_attr, nu, temp_field)

    temp_field = field + dt * delta_field3
    delta_field4 = burgers_graph(edge_list, edge_attr, nu, temp_field)

    delta_field = (
        delta_field1 + 2.0 * delta_field2 + 2.0 * delta_field3 + delta_field4
    ) / 6.0
    return field + dt * delta_field


def solve(
    solver_name: Literal["rk1_graph", "rk2_graph", "rk4_graph"],
    edge_attr: arr,
    edge_list: npt.NDArray[np.int64],
    nu: tuple[float, float],
    initial_field: arr,
    dts: arr,
    device: torch.device,
) -> arr:
    """
    solver_name: How to solve\\
    edge_attr: [Ny, Nx, 2] distance between grid points\\
    edge_list: [E, 2] =[2*Nx*Ny, 2], directed\\
    edge_attr: [E, 2] = [2*Nx*Ny, 2]\\
        [i, 0]: distance of i-th edge. Nonzero if i-th edge is x-axis\\
        [i, 1]: distance of i-th edge. Nonzero if i-th edge is y-axis\\
    nu: [2, ] diffusion coefficient\\
    initial_field: [Ny, Nx, 2]\\
    dts: [S, 1]

    Return
    trajectory: [S+1, N, 2] = [S+1, Nx*Ny, 2]
    """
    rk: Callable[[torch.Tensor, float], torch.Tensor]
    if "rk1" in solver_name:
        rk = functools.partial(
            rk1,
            torch.tensor(edge_list, device=device),
            torch.tensor(edge_attr, device=device),
            torch.tensor(nu, device=device),
        )
    elif "rk2" in solver_name:
        rk = functools.partial(
            rk2,
            torch.tensor(edge_list, device=device),
            torch.tensor(edge_attr, device=device),
            torch.tensor(nu, device=device),
        )
    else:
        rk = functools.partial(
            rk4,
            torch.tensor(edge_list, device=device),
            torch.tensor(edge_attr, device=device),
            torch.tensor(nu, device=device),
        )

    field = torch.tensor(grid2node(initial_field), device=device)
    trajectory = torch.stack([torch.empty_like(field)] * (len(dts) + 1))
    trajectory[0] = field

    for step, dt in enumerate(dts):
        field = rk(field, float(dt))
        trajectory[step + 1] = field

    return trajectory.cpu().numpy()
