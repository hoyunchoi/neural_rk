from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


class StandardScaler(nn.Module):
    def __init__(self, state_dim: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(state_dim, state_dim)

    def fit(self, x: npt.NDArray[np.float32] | torch.Tensor) -> StandardScaler:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        std, avg = torch.std_mean(x, dim=0, unbiased=False)

        self.linear.weight = nn.parameter.Parameter(
            torch.diag(1.0 / std), requires_grad=False
        )
        self.linear.bias = nn.parameter.Parameter(-avg / std, requires_grad=False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class InverseStandardScaler(nn.Module):
    def __init__(self, state_dim: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(state_dim, state_dim)

    def fit(self, x: npt.NDArray[np.float32] | torch.Tensor) -> InverseStandardScaler:
        self.linear = nn.Linear(x.shape[1], x.shape[1])
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        std, avg = torch.std_mean(x, dim=0, unbiased=False)

        self.linear.weight = nn.parameter.Parameter(
            torch.diag(std), requires_grad=False
        )
        self.linear.bias = nn.parameter.Parameter(avg, requires_grad=False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

