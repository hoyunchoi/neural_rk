from __future__ import annotations

import torch
import torch.nn as nn

from neural_rk.protocol import ScalerProtocol
from neural_rk.scaler import IdentityScaler

from .mlp import MLP


class Decoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        node_feature_dim: int,
        bn_momentum: float = 1.0,
        dropout: float = 0.0,
        scaler: ScalerProtocol = IdentityScaler(),
    ) -> None:
        """
        node_embedding_dim: dimension for encoded node features
        node_feature_dim: dimension for decoded node features
        bn_momentum: Refer get_batch_norm_layer
        scaler: inverse transform the decoded value
        """
        super().__init__()
        self.scaler = scaler

        self.node_decoder = MLP(
            node_embedding_dim,
            node_embedding_dim,
            node_feature_dim,
            bn_momentum,
            dropout,
            last=True,
        )

    def forward(self, node_attr: torch.Tensor) -> torch.Tensor:
        """node_attr: [BN, node_emb], embedding of nodes"""
        return self.scaler(self.node_decoder(node_attr))

