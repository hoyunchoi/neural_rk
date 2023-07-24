import colorsys

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

arr = npt.NDArray[np.float32]


def scale_color(color: str, scale: float = 1.0) -> tuple[float, float, float]:
    """
    Change the brightness of given color according to the scale.

    Args
    color: Base color to adjust it's brightness
    scale: number between (0, 2). As scale increases, color becomes brighter

    Return
    color: rgb format color
    """
    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(color))
    return colorsys.hls_to_rgb(h, max(0, min(1, l * scale)), s)


def plot(
    ax1: plt.Axes,
    ax2: plt.Axes,
    ax3: plt.Axes,
    trajectory: arr,
    time: arr | None = None,
    nodes: int | list[int] | npt.NDArray[np.int64] | None = None,
    **kwargs,
) -> None:
    """
    trajectory: [S+1, N, 3]
    time: [S+1, ]
    nodes: which nodes to be drawn
    """
    trajectory = trajectory.transpose(1, 2, 0)  # [N, 3, S+1]

    num_nodes, num_steps = trajectory.shape[0], trajectory.shape[2]
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

    for i, trajectory in enumerate(trajectory[node_idx]):
        if "color" in kwargs:
            color_r = color_g = color_b = kwargs.pop("color")
        else:
            color_scale = 2.0 * (i + 1) / (len(node_idx) + 1)
            color_r = scale_color("red", color_scale)
            color_g = scale_color("green", color_scale)
            color_b = scale_color("blue", color_scale)

        # trajectory of x, y, z position with color r,g,b
        ax1.plot(time, trajectory[0], color=color_r, **kwargs)
        ax2.plot(time, trajectory[1], color=color_g, **kwargs)
        ax3.plot(time, trajectory[2], color=color_b, **kwargs)

    ax1.set_ylabel("x")
    ax2.set_ylabel("y")
    ax3.set_ylabel("z")
