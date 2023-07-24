import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D

from neural_rk.grid_graph import to_periodic_field

arr = npt.NDArray[np.float32]


def plot_1d_image(
    ax: plt.Axes,
    time: arr,
    position: arr,
    trajectory: arr,
    lim: tuple[float, float] = (-1.0, 1.0),
    to_periodic: bool = True,
    **kwargs
) -> ScalarMappable:
    """
    time: [S+1, ]
    position: [Nx+1, 1]
    trajectory: if to_periodic is True, [S+1, Nx, 1], otherwise [S+1, Nx+1, 1]
    """
    if to_periodic:
        trajectory = np.stack([to_periodic_field(field) for field in trajectory], axis=0) # [S+1, Nx+1, 1]

    trajectory = trajectory.squeeze().T # [Nx+1, S+1]
    time_pos = np.stack(np.meshgrid(time, position.squeeze()), axis=0)  # [2, Nx+1, S+1]

    heatmap = ax.pcolormesh(
        *time_pos, trajectory, cmap="coolwarm", vmin=lim[0], vmax=lim[1], **kwargs
    )

    return heatmap

def plot_1d(
    ax: plt.Axes,
    position: arr,
    field: arr,
    lim: tuple[float, float] = (-1.0, 1.0),
    to_periodic: bool = True,
) -> ScalarMappable:
    """
    position: [Nx+1, 1]
    trajectory: if to_periodic is True, [Nx, 1], otherwise [Nx+1, 1]
    """
    if to_periodic:
        field = to_periodic_field(field)

    u = field[..., 0]
    sc = ax.scatter(position[..., 0], u, c=u, cmap="coolwarm", vmin=lim[0], vmax=lim[1])
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_xlim(position.min(), position.max())
    ax.set_ylim(*lim)
    return sc



def plot_2d(
    ax1: Axes3D,  # type: ignore
    ax2: Axes3D,  # type: ignore
    position: arr,
    field: arr,
    lim: tuple[float, float] = (-1.0, 1.0),
    to_periodic: bool = True,
) -> tuple[ScalarMappable, ScalarMappable]:
    """
    position: [Ny+1, Nx+1, 2]
    trajectory: if to_periodic is True, [Ny, Nx, 2], otherwise [Ny+1, Nx+1, 2]
    """
    if to_periodic:
        field = to_periodic_field(field)
    x, y = position[..., 0], position[..., 1]
    sf1 = ax1.plot_surface(
        x, y, field[..., 0], cmap="coolwarm", vmin=lim[0], vmax=lim[1]
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u")
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(y.min(), y.max())
    ax1.set_zlim(lim)

    sf2 = ax2.plot_surface(
        x, y, field[..., 1], cmap="coolwarm", vmin=lim[0], vmax=lim[1]
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("v")
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(y.min(), y.max())
    ax2.set_zlim(lim)

    return sf1, sf2
