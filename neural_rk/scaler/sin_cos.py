from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


class SinCosScaler(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()

    def fit(self, x: npt.NDArray[np.float32] | torch.Tensor) -> SinCosScaler:
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat((x.sin(), x.cos()), dim=-1)

