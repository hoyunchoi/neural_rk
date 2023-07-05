import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.image import AxesImage


def plot(
    ax: plt.Axes,
    trajectories: npt.NDArray[np.float32],
    nodes: int | npt.NDArray[np.int64] | None = None,
    xticks: npt.NDArray[np.float32] | None = None,
    **kwargs,
) -> AxesImage:
    """
    trajectories: [S+1, N, 1]
    nodes: which nodes to be drawn
    """
    trajectories = trajectories.squeeze()

    num_steps, num_nodes = trajectories.shape
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

    im = ax.imshow(trajectories.T[node_idx], aspect="auto", **kwargs)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, num_steps, len(xticks)))
    ax.set_xticklabels(xticks)
    return im
