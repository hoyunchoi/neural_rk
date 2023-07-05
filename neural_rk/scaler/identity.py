from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


class IdentityScaler(nn.Module):
    def __init__(self, state_dim: int = 0) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def fit(self, x: npt.NDArray[np.float32] | torch.Tensor) -> IdentityScaler:
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.identity(x)
