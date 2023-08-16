import functools
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt

from neural_rk.grid_graph import grid2node

arr = npt.NDArray[np.float32]


def burgers_2d(dx: arr, dy: arr, nu: tuple[float, float], field: arr) -> arr:
    """
    dx: [Nx-1, ] distance between grid points at x-axis
    dy: [Ny-1, ] distance between grid points at y-axis
    dx, dy: [Ny-1, Nx-1, 2] distance between grid points
    nu: [2, ] diffusion coefficient
    field: [Ny, Nx, 2], xy-field of each grid point

    Return: [Ny, Nx, 2], delta field
    """
    # [Ny, Nx], [Ny, Nx]
    u, v = field[..., 0], field[..., 1]
    nu_x, nu_y = nu
    dx, dy = dx[None, :], dy[:, None]  # [1, Nx-1], [Ny-1, 1]
    dx_double = dx[:, 1:] + dx[:, :-1]  # [1, Nx-2 ]
    dy_double = dy[1:, :] + dy[:-1, :]  # [Ny-2, 1]

    # Half derivatives, handle like uniform grid
    u_x_half = (u[:, 1:] - u[:, :-1]) / dx  # [Ny, Nx-1]
    u_y_half = (u[1:, :] - u[:-1, :]) / dy  # [Ny-1, Nx]
    v_x_half = (v[:, 1:] - v[:, :-1]) / dx  # [Ny, Nx-1]
    v_y_half = (v[1:, :] - v[:-1, :]) / dy  # [Ny-1, Nx]

    # Derivatives: Average of half derivatives
    u_x = (  # [Ny, Nx-2]
        dx[:, 1:] * u_x_half[:, :-1] + dx[:, :-1] * u_x_half[:, 1:]
    ) / dx_double
    u_y = (  # [Ny-2, Nx]
        dy[1:, :] * u_y_half[:-1, :] + dy[:-1, :] * u_y_half[1:, :]
    ) / dy_double
    v_x = (  # [Ny, Nx-2]
        dx[:, 1:] * v_x_half[:, :-1] + dx[:, :-1] * v_x_half[:, 1:]
    ) / dx_double
    v_y = (  # [Ny-2, Nx]
        dy[1:, :] * v_y_half[:-1, :] + dy[:-1, :] * v_y_half[1:, :]
    ) / dy_double

    u_x = np.pad(u_x, [(0, 0), (1, 1)])  # [Ny, Nx]
    u_y = np.pad(u_y, [(1, 1), (0, 0)])  # [Ny, Nx]
    v_x = np.pad(v_x, [(0, 0), (1, 1)])  # [Ny, Nx]
    v_y = np.pad(v_y, [(1, 1), (0, 0)])  # [Ny, Nx]

    # Second derivatives: differentiation over half derivatives
    u_xx = 2.0 * (u_x_half[:, 1:] - u_x_half[:, :-1]) / dx_double  # [Ny, Nx-2]
    u_yy = 2.0 * (u_y_half[1:, :] - u_y_half[:-1, :]) / dy_double  # [Ny-2, Nx]
    v_xx = 2.0 * (v_x_half[:, 1:] - v_x_half[:, :-1]) / dx_double  # [Ny, Nx-2]
    v_yy = 2.0 * (v_y_half[1:, :] - v_y_half[:-1, :]) / dy_double  # [Ny-2, Nx]
    u_xx = np.pad(u_xx, [(0, 0), (1, 1)])  # [Ny, Nx]
    u_yy = np.pad(u_yy, [(1, 1), (0, 0)])  # [Ny, Nx]
    v_xx = np.pad(v_xx, [(0, 0), (1, 1)])  # [Ny, Nx]
    v_yy = np.pad(v_yy, [(1, 1), (0, 0)])  # [Ny, Nx]

    # Only remain non-boundary values
    delta = np.stack(
        (
            -u * u_x - v * u_y + nu_x * u_xx + nu_y * u_yy,
            -u * v_x - v * v_y + nu_x * v_xx + nu_y * v_yy,
        ),
        axis=-1,
    )
    delta[(0, -1), :] = 0.0
    delta[:, (0, -1)] = 0.0
    return delta


def rk1(dx: arr, dy: arr, nu: tuple[float, float], field: arr, dt: float) -> arr:
    """
    dx, dy: [Ny, Nx, 2] distance between grid points
    nu: [2, ] diffusion coefficient
    field: [Ny, Nx, 2], xy-field of each grid point
    dt

    Return: [Ny, Nx, 2] next field
    """
    delta_field = burgers_2d(dx, dy, nu, field)
    return field + dt * delta_field


def rk2(dx: arr, dy: arr, nu: tuple[float, float], field: arr, dt: float) -> arr:
    """
    dx, dy: [Ny, Nx, 2] distance between grid points
    nu: [2, ] diffusion coefficient
    field: [Ny, Nx, 2], xy-field of each grid point
    dt

    Return: [Ny, Nx, 2] next field
    """
    delta_field1 = burgers_2d(dx, dy, nu, field)

    temp_field = field + dt * delta_field1
    delta_field2 = burgers_2d(dx, dy, nu, temp_field)

    delta_field = 0.5 * (delta_field1 + delta_field2)
    return field + dt * delta_field


def rk4(dx: arr, dy: arr, nu: tuple[float, float], field: arr, dt: float) -> arr:
    """
    dx, dy: [Ny, Nx, 2] distance between grid points
    nu: [2, ] diffusion coefficient
    field: [Ny, Nx, 2], xy-field of each grid point
    dt

    Return: [Ny, Nx, 2] next field
    """
    delta_field1 = burgers_2d(dx, dy, nu, field)

    temp_field = field + 0.5 * dt * delta_field1
    delta_field2 = burgers_2d(dx, dy, nu, temp_field)

    temp_field = field + 0.5 * dt * delta_field2
    delta_field3 = burgers_2d(dx, dy, nu, temp_field)

    temp_field = field + dt * delta_field3
    delta_field4 = burgers_2d(dx, dy, nu, temp_field)

    delta_field = (
        delta_field1 + 2.0 * delta_field2 + 2.0 * delta_field3 + delta_field4
    ) / 6.0
    return field + dt * delta_field


def solve(
    solver_name: Literal["rk1", "rk2", "rk4"],
    dxdy: tuple[arr, arr],
    nu: tuple[float, float],
    field: arr,
    dts: arr,
) -> arr:
    """
    solver_name: How to solve \\
    dxdy: [Nx-1, ], [Ny-1, ] distance between grid points \\
    nu: [2, ] diffusion coefficient \\
    initial_field: [Ny, Nx, 2] \\
    dts: [S, 1]

    Return
    trajectory: [S+1, N, 2] = [S+1, Nx*Ny, 2]
    """
    rk: Callable[[arr, float], arr]
    if "rk1" == solver_name:
        rk = functools.partial(rk1, *dxdy, nu)
    elif "rk2" == solver_name:
        rk = functools.partial(rk2, *dxdy, nu)
    else:
        rk = functools.partial(rk4, *dxdy, nu)

    trajectory = np.stack([np.empty_like(field).reshape(-1, 2)] * (len(dts) + 1))
    trajectory[0] = grid2node(field)
    for step, dt in enumerate(dts):
        try:
            field = rk(field, dt)
            trajectory[step + 1] = grid2node(field)
        except FloatingPointError:
            # When field diverge: stop iteration and return nan
            return np.array([np.nan])

    return trajectory
