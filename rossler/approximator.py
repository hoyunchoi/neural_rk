from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from neural_rk.hyperparameter import ApproximatorParameter
from neural_rk.modules import MLP, Decoder, Encoder
from neural_rk.protocol import ApproximatorProtocol, ScalerProtocol
from neural_rk.scaler import get_scaler


class RosslerApproximator(nn.Module):
    def __init__(
        self,
        state_embedding_dims: list[int],
        node_embedding_dims: list[int],  # Empty
        edge_embedding_dims: list[int],
        glob_embedding_dims: list[int],
        node_hidden_dim: int,
        edge_hidden_dim: int,
        scalers: tuple[ScalerProtocol, ScalerProtocol],
        dropout: float,
        bn_momentum: float,
    ) -> None:
        super().__init__()
        state_dim = len(state_embedding_dims)
        node_emb_dim = sum(state_embedding_dims)
        edge_emb_dim = sum(edge_embedding_dims)
        glob_emb_dim = sum(glob_embedding_dims)

        self.state_encoder = Encoder(
            node_embedding_dims=state_embedding_dims,
            bn_momentum=bn_momentum,
            dropout=dropout,
            scaler=scalers[0],
        )
        self.encoder = Encoder(
            node_embedding_dims,
            edge_embedding_dims,
            glob_embedding_dims,
            bn_momentum,
            dropout,
        )

        self.edge_calculator = MLP(
            2 * node_emb_dim + edge_emb_dim,
            edge_hidden_dim,
            edge_emb_dim,
            bn_momentum,
            dropout
        )
        self.node_calculator = MLP(
            edge_emb_dim + glob_emb_dim,
            node_hidden_dim,
            node_emb_dim,
            bn_momentum,
            dropout
        )

        self.state_decoder = Decoder(
            node_emb_dim, state_dim, bn_momentum, dropout, scaler=scalers[1]
        )

    @classmethod
    def from_hp(
        cls,
        hp: ApproximatorParameter,
        scalers: tuple[ScalerProtocol, ScalerProtocol] | None = None,
    ) -> ApproximatorProtocol:
        if scalers is None:
            scalers = (
                get_scaler("rossler", hp.in_scaler, inverse=False),
                get_scaler("rossler", hp.out_scaler, inverse=True),
            )

        return cls(
            hp.state_embedding,
            hp.node_embedding,
            hp.edge_embedding,
            hp.glob_embedding,
            hp.node_hidden,
            hp.edge_hidden,
            scalers,
            hp.dropout,
            hp.bn_momentum,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        glob_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (BN, 3), state_dim=len(state_embedding_dims), State of each node
        edge_index: (2, BE),
        batch: (BN, ), index of batch where each node belongs

        node_attr: (N, 0), node_attr_dim=len(node_embedding_dims), Parameters of each node
        edge_attr: (BE, 1), edge_attr_dim=len(edge_embedding_dims), Parameters of each edge
        glob_attr: (B, 3), glob_attr_dim=len(glob_embedding_dims), Parameters of each glob
        """
        row, col = edge_index

        # Encoding: (BN, state_emb), (BE, edge_emb), (B, glob_emb)
        node_emb, *_ = self.state_encoder(x)
        _, edge_emb, glob_emb = self.encoder(node_attr, edge_attr, glob_attr)

        # Edge aggregation -> Edge update
        # (BE, 2*node_emb + edge_emb) -> (BE, edge_emb)
        interaction = torch.cat((node_emb[row], node_emb[col], edge_emb), dim=-1)
        interaction = self.edge_calculator(interaction)

        # Node sum aggregation -> node update
        # (BE, edge_emb) -> (BN, edge_emb) -> (BN, edge_emb + glob_emb) -> (BN, node_emb)
        aggregated = scatter_sum(interaction, col, dim=0, dim_size=len(batch))
        aggregated = torch.cat((aggregated, glob_emb[batch]), dim=1)
        aggregated = self.node_calculator(aggregated)

        # Decoding: (BN, node_emb) -> (BN, state_dim)
        return self.state_decoder(aggregated)
