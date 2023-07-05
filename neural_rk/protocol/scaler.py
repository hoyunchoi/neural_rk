from __future__ import annotations

from typing import Any, Mapping, Protocol, overload

import numpy as np
import numpy.typing as npt
import torch
from torch.nn.modules.module import _IncompatibleKeys


class ScalerProtocol(Protocol):
    def __init__(self, state_dim: int) -> None:
        ...

    def fit(self, x: npt.NDArray[np.float32] | torch.Tensor) -> ScalerProtocol:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @overload
    def state_dict(
        self, *, destination: dict[str, Any], prefix: str = ..., keep_vars: bool = ...
    ) -> dict[str, Any]:
        ...

    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> dict[str, Any]:
        ...

    def state_dict(
        self, *args, destination=None, prefix="", keep_vars=False
    ) -> dict[str, Any]:
        ...

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ) -> _IncompatibleKeys:
        ...
