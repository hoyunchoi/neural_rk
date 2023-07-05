from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


class MinMaxScaler(nn.Module):
    def __init__(self, state_dim: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(state_dim, state_dim)

    def fit(self, x: npt.NDArray[np.float32] | torch.Tensor) -> MinMaxScaler:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        maximum, minimum = x.amax(dim=0), x.amin(dim=0)

        self.linear.weight = nn.parameter.Parameter(
            torch.diag(1.0 / (maximum - minimum)), requires_grad=False
        )
        self.linear.bias = nn.parameter.Parameter(
            -minimum / (maximum - minimum), requires_grad=False
        )
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class InverseMinMaxScaler(nn.Module):
    def __init__(self, state_dim: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(state_dim, state_dim)

    def fit(self, x: npt.NDArray[np.float32] | torch.Tensor) -> InverseMinMaxScaler:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        maximum, minimum = x.amax(dim=0), x.amin(dim=0)

        self.linear.weight = nn.parameter.Parameter(
            torch.diag(maximum - minimum), requires_grad=False
        )
        self.linear.bias = nn.parameter.Parameter(minimum, requires_grad=False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
