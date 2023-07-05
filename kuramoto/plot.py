import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.image import AxesImage

from .trajectory import normalize_trajectory


def plot_trajectory(
    ax: plt.Axes,
    trajectory: npt.NDArray[np.float32],
    nodes: int | npt.NDArray[np.int64] | None = None,
    xticks: npt.NDArray[np.float32] | None = None,
    **kwargs,
) -> AxesImage:
    """
    trajectory: [S+1, N, 1]
    nodes: which nodes to be drawn
    """
    trajectory = normalize_trajectory(trajectory.squeeze())

    num_steps, num_nodes = trajectory.shape
    if isinstance(nodes, int):
        node_idx = np.random.choice(num_nodes, nodes, replace=False)
    elif isinstance(nodes, list):
        node_idx = np.array(nodes, dtype=np.int64)
    elif isinstance(nodes, np.ndarray):
        node_idx = nodes.astype(np.int64, copy=False)
    else:
        node_idx = np.arange(num_nodes)

    if xticks is None:
        xticks = np.linspace(0, num_steps, num=5, dtype=np.float32)

    im = ax.imshow(
        trajectory.T[node_idx], interpolation="none", aspect="auto", **kwargs
    )
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, num_steps, len(xticks)))
    ax.set_xticklabels(xticks)

    return im
