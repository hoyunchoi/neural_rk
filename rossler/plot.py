import colorsys

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot(
    ax1: plt.Axes,
    ax2: plt.Axes,
    ax3: plt.Axes,
    trajectories: npt.NDArray[np.float32],
    time: npt.NDArray[np.float32] | None = None,
    nodes: int | list[int] | npt.NDArray[np.int64] | None = None,
    **kwargs,
) -> None:
    """
    trajectories: [S, N, 3]
    nodes: which nodes to be drawn
    """
    def scale_color(color: str, scale: float = 1.0) -> tuple[float, float, float]:
        """0 < scale < 2
        As scale increases, color becomes brighter"""
        h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(color))
        return colorsys.hls_to_rgb(h, max(0, min(1, l * scale)), s)

    num_steps, num_nodes = trajectories.shape[0], trajectories.shape[1]
    if isinstance(nodes, int):
        node_idx = np.random.choice(num_nodes, nodes, replace=False)
    elif isinstance(nodes, list):
        node_idx = np.array(nodes, dtype=np.int64)
    elif isinstance(nodes, np.ndarray):
        node_idx = nodes.astype(np.int64, copy=False)
    else:
        node_idx = np.arange(num_nodes)

    if time is None:
        time = np.arange(num_steps, dtype=np.float32)

    for i, trajectory in enumerate(trajectories[:, node_idx, :].transpose(1, 2, 0)):
        if "color" in kwargs:
            color_r = color_g = color_b = kwargs.pop("color")
        else:
            color_scale = 2.0 * (i + 1) / (len(node_idx) + 1)
            color_r = scale_color("red", color_scale)
            color_g = scale_color("green", color_scale)
            color_b = scale_color("blue", color_scale)

        # position: trajectory
        ax1.plot(time, trajectory[0], color=color_r, **kwargs)

        # position: y
        ax2.plot(time, trajectory[1], color=color_g, **kwargs)

        # position: z
        ax3.plot(time, trajectory[2], color=color_b, **kwargs)
    ax1.set_ylabel("x")
    ax2.set_ylabel("y")
    ax3.set_ylabel("z")
