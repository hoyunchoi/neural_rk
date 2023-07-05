from typing import cast

import numpy as np
import numpy.typing as npt
import torch

TOLERANCE = 2.0 * np.pi


class IsDiverging:
    tolerance = TOLERANCE

    def __call__(self, trajectory: torch.Tensor) -> torch.BoolTensor:
        return cast(torch.BoolTensor, trajectory.isnan().any())


class IsDivergingPrecise:
    tolerance = TOLERANCE

    def __call__(self, trajectory: torch.Tensor) -> torch.BoolTensor:
        nan_indices = trajectory.isnan().nonzero()
        if nan_indices.numel():
            print(f"Nan detected at {nan_indices.cpu().squeeze()}", end=" ")
            return cast(torch.BoolTensor, torch.tensor(True, dtype=torch.bool))

        return cast(torch.BoolTensor, torch.tensor(False, dtype=torch.bool))


def normalize_trajectory(
    trajectory: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    return ((trajectory + np.pi) % (2 * np.pi) - np.pi).view(trajectory.dtype)


def order_parameter(
    trajectory: npt.NDArray[np.float32] | torch.Tensor,
) -> npt.NDArray[np.float32]:
    """
    trajectory: [S, N, 1] or [N, 1]
    return: [S, ] or [1, ]
    """
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()

    trajectory = trajectory.squeeze()
    return np.abs(np.exp(1.0j * trajectory).mean(axis=-1))


def compare_trajectory(
    trajectory1: npt.NDArray[np.float32] | torch.Tensor,
    trajectory2: npt.NDArray[np.float32] | torch.Tensor,
    log: bool = True,
) -> npt.NDArray[np.float32]:
    """
    trajectory1: (S+1, N, 1)
    trajectory2: (S+1, N, 1)

    return: (S, ), MAE of each time step, averaged over nodes
    """
    if isinstance(trajectory1, torch.Tensor):
        trajectory1 = trajectory1.cpu().numpy()
    if isinstance(trajectory2, torch.Tensor):
        trajectory2 = trajectory2.cpu().numpy()
    trajectory1 = cast(npt.NDArray[np.float32], trajectory1.squeeze())
    trajectory2 = cast(npt.NDArray[np.float32], trajectory2.squeeze())

    abs_err = np.abs(trajectory2 - trajectory1)
    if log:
        max_err_idx = np.unravel_index(np.argmax(abs_err), abs_err.shape)
        step, node = max_err_idx
        print(
            f"MAE={abs_err.mean():.4f}, "
            f"Maximum err: {abs_err[max_err_idx]:.4f} at {step=}, {node=}"
        )
    return np.mean(abs_err, axis=1)
