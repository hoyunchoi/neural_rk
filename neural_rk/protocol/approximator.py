from __future__ import annotations

from typing import Any, Protocol

import torch
from torch.nn.modules.module import _IncompatibleKeys

from neural_rk.hyperparameter import ApproximatorParameter

from .scaler import ScalerProtocol


class ApproximatorProtocol(Protocol):
    def __init__(
        self,
        state_embedding_dims: list[int],
        node_embedding_dims: list[int],
        edge_embedding_dims: list[int],
        glob_embedding_dims: list[int],
        node_hidden_dim: int,
        edge_hidden_dim: int,
        scalers: tuple[ScalerProtocol, ScalerProtocol],
        dropout: float,
        bn_momentum: float,
    ) -> None:
        ...

    @classmethod
    def from_hp(
        cls,
        hp: ApproximatorParameter,
        scalers: tuple[ScalerProtocol, ScalerProtocol] | None = None,
    ) -> ApproximatorProtocol:
        ...

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        glob_attr: torch.Tensor,
    ) -> torch.Tensor:
        ...

    # ------------------- torch.nn.Module --------------------
    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        glob_attr: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> _IncompatibleKeys:
        ...
