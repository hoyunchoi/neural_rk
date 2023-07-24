import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.image import AxesImage

from .trajectory import normalize_trajectory

arr = npt.NDArray[np.float32]


def plot(
    ax: plt.Axes,
    trajectory: arr,
    time: arr | None = None,
    nodes: int | npt.NDArray[np.int64] | None = None,
    lim: tuple[float, float] = (-np.pi, np.pi),
    **kwargs,
) -> None:
    """
    trajectory: [S+1, N, 1]
    time: [S+1, ]
    nodes: which nodes to be drawn
    """
    trajectory = normalize_trajectory(trajectory.squeeze().T)  # [N, S+1]

    num_nodes, num_steps = trajectory.shape
    if time is None:
        time = np.arange(num_steps, dtype=np.float32)

    if isinstance(nodes, int):
        node_idx = np.random.choice(num_nodes, nodes, replace=False)
    elif isinstance(nodes, list):
        node_idx = np.array(nodes, dtype=np.int64)
    elif isinstance(nodes, np.ndarray):
        node_idx = nodes.astype(np.int64, copy=False)
    else:
        node_idx = np.arange(num_nodes)

    for node in node_idx:
        ax.scatter(
            time,
            trajectory[node],
            c=trajectory[node],
            s=2,
            cmap="twilight_shifted",
            vmin=min(lim),
            vmax=max(lim),
            **kwargs,
        )


def plot_image(
    ax: plt.Axes,
    trajectory: arr,
    nodes: int | npt.NDArray[np.int64] | None = None,
    lim: tuple[float, float] = (-np.pi, np.pi),
    xticks: arr | None = None,
    **kwargs,
) -> AxesImage:
    """
    trajectory: [S+1, N, 1]
    nodes: which nodes to be drawn
    """
    trajectory = normalize_trajectory(trajectory.squeeze().T)  # [N, S+1]

    num_nodes, num_steps = trajectory.shape
    if isinstance(nodes, int):
        node_idx = np.random.choice(num_nodes, nodes, replace=False)
    elif isinstance(nodes, list):
        node_idx = np.array(nodes, dtype=np.int64)
    elif isinstance(nodes, np.ndarray):
        node_idx = nodes.astype(np.int64, copy=False)
    else:
        node_idx = np.arange(num_nodes)

    im = ax.imshow(
        trajectory[node_idx],
        interpolation="none",
        aspect="auto",
        cmap="twilight_shifted",
        vmin=min(lim),
        vmax=max(lim),
        **kwargs,
    )

    if xticks is None:
        xticks = np.linspace(0, num_steps, num=5, dtype=np.float32)
    ax.set_xticks(np.linspace(0, num_steps, len(xticks)))
    ax.set_xticklabels(xticks)
    ax.set_yticks([])

    return im
