import numpy as np
import pandas as pd
import torch

from neural_rk.grid_graph_dirichlet import edge2dxdy, node2grid


def preprocess(data: pd.Series) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract mesh with time and field from data.

    Result
    xyt: [Ny+1, Nx+1, S+1, 3], xyt value
    field: [S+1, Ny+1, Nx+1, 2], uv value
    nu: [2,] nu, nu
    """
    # Extract Nx, Ny
    Nx, Ny = tuple(map(int, data.network_type.split("_")))

    # Extract dx, dy: spacing of x-axis and y-axis
    dx, dy = edge2dxdy(data.edge_attr, Nx, Ny)  # [Nx-1, ], [Ny-1, ]
    x = np.insert(np.cumsum(dx), 0, 0.0)  # [Nx, ]
    y = np.insert(np.cumsum(dy), 0, 0.0)  # [Ny, ]

    # Extract time of shape [S+1, ]
    time = np.insert(np.cumsum(data.dts.numpy()), 0, 0.0)

    # Mesh of x, y, t [Ny, Nx, S+1, 3]
    xyt = np.stack(np.meshgrid(x, y, time), axis=-1)

    # Convert field to grid shape field
    # node2grid: [N, 2] -> [Ny, Nx, 2]
    # field: [S+1, Ny, Nx, 2]
    field = np.stack(
        [node2grid(traj, Nx, Ny) for traj in data.trajectories],
        axis=0,
    )

    # Extract nu of shape [2,]
    nu = data.glob_attr.numpy().squeeze()

    # [Ny+1, Nx+1, S+1, 3], [S+1, Ny+1, Nx+1]
    return torch.tensor(xyt), torch.tensor(field), torch.as_tensor(nu)