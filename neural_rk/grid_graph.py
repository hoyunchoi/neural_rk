from typing import cast

import numpy as np
import numpy.typing as npt
import torch

arr = npt.NDArray[np.float32]


def to_periodic_field(field: arr) -> arr:
    """
    Make field to be periodic
    [Nx, 1] -> [Nx+1, 1]
    [Ny, Nx, 2] -> [Ny+1, Nx+1, 2]
    """
    if field.ndim == 2:
        # [Nx, 1] -> [Nx+1, 1]
        assert field.shape[1] == 1

        pad = [(0, 1), (0, 0)]
        periodic_field = np.pad(field, pad)
        periodic_field[-1, :] = field[0, :]

        return periodic_field

    elif field.ndim == 3:
        # [Ny, Nx, 2] -> [Ny+1, Nx+1, 2]
        pad = [(0, 1), (0, 1), (0, 0)]
        periodic_field = np.pad(field, pad)

        periodic_field[-1, :-1] = field[0, :]
        periodic_field[:-1, -1] = field[:, 0]
        periodic_field[-1, -1] = field[0, 0]

        return periodic_field

    raise ValueError


def grid2node(node_attr: arr) -> arr:
    """
    Grid point to node point. Node always corresponds to 2D \\
    Inverse of node2grid

    [Nx, 1] -> [Nx, 2] \\
    [Ny, Nx, 2] -> [Nx*Ny, 2]
    """
    ndim = node_attr.shape[-1]
    assert node_attr.ndim == ndim + 1

    if ndim == 1:
        # [Nx, 1] -> [Nx, 2]
        return np.pad(node_attr, [(0, 0), (0, 1)])
    elif ndim == 2:
        # [Ny, Nx, 2] -> [Nx*Ny, 2]
        return node_attr.reshape(-1, 2)

    raise ValueError


def node2grid(node_attr: torch.Tensor | arr, *NxNy: int) -> arr:
    """
    Node point to grid point. Dimension of grid point corresponds to len(NxNy) \\
    Inverse of grid2node

    [Nx, 2] -> [Nx, 1] \\
    [Nx*Ny, 2] -> [Ny, Nx, 2] \\
    """
    assert np.prod(NxNy) == len(node_attr)
    ndim = len(NxNy)

    if isinstance(node_attr, torch.Tensor):
        node_attr = cast(arr, node_attr.cpu().numpy())

    if ndim == 1:
        # [Nx, 2] -> [Nx, 1]
        return node_attr[..., :1]
    elif ndim == 2:
        # [Nx*Ny, 2] -> [Ny, Nx, 2]
        Nx, Ny = NxNy
        return node_attr.reshape(Ny, Nx, 2)

    raise ValueError


def dxdy2pos(*dxdy: arr) -> arr:
    """
    From the spacing of each axis, generate positions of all grid points

    [Nx, ] -> [Nx+1, 1] \\
    [Nx, ], [Ny, ] -> [Ny+1, Nx+1, 2] \\
    """
    return np.stack(
        np.meshgrid(*[np.insert(np.cumsum(dx), 0, 0.0) for dx in dxdy]), axis=-1
    )


def dxdy2edge(*dxdy: arr) -> arr:
    """
    From the spacing of each axis, generate edge attributes. \\
    Edge attributes always corresponds to 3D \\
    Inverse of edge2dxdy

    [Nx, ] -> [Nx, 2] = [E, 2] \\
    [Nx, ], [Ny, ] -> [2*Nx*Ny, 2] = [E, 2] \\

    Return
    edge_attr[..., 0]: nonzero for x-axis \\
    edge_attr[..., 1]: nonzero for y-axis \\
    """
    ndim = len(dxdy)

    if ndim == 1:
        # [E, ] -> [E, 1] -> [E, 2]
        (dx,) = dxdy
        edge_attr = dx[..., None]
        return np.pad(edge_attr, ((0, 0), (0, 1)))

    elif ndim == 2:
        dx, dy = dxdy
        Nx, Ny = len(dx), len(dy)

        # [E, 2]
        edge_attr = np.zeros((2 * Nx * Ny, 2), dtype=np.float32)
        edge_attr[::2, 0] = np.tile(dx, Ny)
        edge_attr[1::2, 1] = np.repeat(dy, Nx)
        return edge_attr

    raise ValueError


def edge2dxdy(edge_attr: torch.Tensor | arr, *NxNy: int) -> tuple[arr, ...]:
    """
    From the edge attributes, generate spacing of each axis \\
    Inverse of dxdy2edge

    [E, 2] = [Nx, 2] -> [Nx, ] \\
    [E, 2] = [2*Nx*Ny, 2] -> [Nx, ], [Ny, ] \\
    """
    assert len(edge_attr) % np.prod(NxNy) == 0
    ndim = len(edge_attr) // np.prod(NxNy)
    assert ndim == len(NxNy)

    if isinstance(edge_attr, torch.Tensor):
        edge_attr = cast(arr, edge_attr.cpu().numpy())

    if ndim == 1:
        return (edge_attr[..., 0],)

    elif ndim == 2:
        Nx, _ = NxNy
        return (edge_attr[: 2 * Nx : 2, 0], edge_attr[1 :: 2 * Nx, 1])

    raise ValueError
