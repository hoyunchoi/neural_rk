from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from neural_rk.hyperparameter import ApproximatorParameter
from neural_rk.modules import MLP, Decoder, DuplicatedEncoder
from neural_rk.protocol import ApproximatorProtocol, ScalerProtocol
from neural_rk.scaler import get_scaler


def clean(x: torch.Tensor) -> torch.Tensor:
    """Inf, NaN to zero"""
    return x.nan_to_num(0.0, posinf=0.0, neginf=0.0)


class BurgersApproximator(nn.Module):
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
        state_dim = 2
        node_emb_dim = state_embedding_dims[0]
        edge_emb_dim1 = edge_embedding_dims[0]  # spacing_emb
        edge_emb_dim2 = edge_embedding_dims[1]  # inv_spacing_emb
        glob_emb_dim = glob_embedding_dims[0]

        self.encoder1 = DuplicatedEncoder(
            node_emb_dim,
            edge_emb_dim1,
            glob_emb_dim,
            bn_momentum,
            dropout,
            scaler=scalers[0],
        )
        self.encoder2 = DuplicatedEncoder(
            edge_embedding_dim=edge_emb_dim2,
            bn_momentum=bn_momentum,
            dropout=dropout,
        )
        self.derivative_calculator = MLP(
            4 * node_emb_dim + 2 * edge_emb_dim1,
            edge_hidden_dim,
            2 * edge_emb_dim1,
            bn_momentum,
            dropout,
        )
        self.half_jacobian_calculator = MLP(
            4 * edge_emb_dim1, edge_hidden_dim, 2 * edge_emb_dim2, bn_momentum, dropout
        )
        self.jacobian_calculator = MLP(
            6 * edge_emb_dim2, node_hidden_dim, 2 * node_emb_dim, bn_momentum, dropout
        )
        self.hessian_calculator = MLP(
            4 * edge_emb_dim1 + 2 * edge_emb_dim2,
            node_hidden_dim,
            2 * node_emb_dim,
            bn_momentum,
            dropout,
        )

        self.node_calculator = MLP(
            6 * node_emb_dim + 2 * glob_emb_dim,
            node_hidden_dim,
            2 * node_emb_dim,
            bn_momentum,
            dropout,
        )
        self.state_decoder = Decoder(
            2 * node_emb_dim, state_dim, bn_momentum, dropout, scalers[1]
        )

    @classmethod
    def from_hp(
        cls,
        hp: ApproximatorParameter,
        scalers: tuple[ScalerProtocol, ScalerProtocol] | None = None,
    ) -> ApproximatorProtocol:
        if scalers is None:
            scalers = (
                get_scaler("burgers", hp.in_scaler, inverse=False),
                get_scaler("burgers", hp.out_scaler, inverse=True),
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
        x: [BN, 2], State of each node
        edge_index: [2, BE],
        batch: [BN, ], index of batch where each node belongs

        node_attr: [BN, 2], If this data is 1D or 2D
        edge_attr: [BE, 2], spacing of edges
        glob_attr: [B, 2], nu
        """
        row, col = edge_index
        spacing = edge_attr  # [BE, 2]
        inv_spacing = clean(1.0 / spacing)  # [BE, 2]
        inv_double_spacing = clean(  # [BN, 2]
            1.0 / (scatter_sum(spacing, row, dim=0) + scatter_sum(spacing, col, dim=0))
        )

        # Encoding: [BN, 2*node_emb], [BE, 2*edge_emb1], [B, 2*glob_emb]
        node_emb, spacing_emb, glob_emb = self.encoder1(x, spacing, glob_attr)
        # [BE, 2*edge_emb2], [BN, 2*edge_emb2]
        _, inv_spacing_emb, _ = self.encoder2(edge_attr=inv_spacing)
        _, inv_double_spacing_emb, _ = self.encoder2(edge_attr=inv_double_spacing)

        # Half derivatives
        # [BE, 4*node_emb + 2*edge_emb1] -> [BE, 2*edge_emb1] -> [BN, 2*edge_emb1], [BN, 2*edge_emb1]
        derivative_per_edge = torch.cat(
            (node_emb[row], node_emb[col], inv_spacing_emb), dim=-1
        )
        derivative_per_edge = self.derivative_calculator(derivative_per_edge)
        half_jacobian_plus = scatter_sum(derivative_per_edge, row, dim=0)
        half_jacobian_minus = scatter_sum(derivative_per_edge, col, dim=0)

        # Jacobian
        # [BE, 2*edge_emb1 + 2*edge_emb1] -> [BE, 2 * edge_emb2] -> [BN, 2*edge_emb2]
        jacobian1 = self.half_jacobian_calculator(
            torch.cat((spacing_emb, half_jacobian_plus[col]), dim=-1)
        )
        jacobian1 = scatter_sum(jacobian1, col, dim=0)

        # [BE, 2*edge_emb1 + 2*edge_emb1] -> [BE, 2 * edge_emb2] -> [BN, 2*edge_emb2]
        jacobian2 = self.half_jacobian_calculator(
            torch.cat((spacing_emb, half_jacobian_minus[row]), dim=-1)
        )
        jacobian2 = scatter_sum(jacobian2, row, dim=0)

        # [BN, 2*edge_emb2 + 2*edge_emb2 + 2*edge_emb2] -> [BN, 2*node_emb]
        jacobian = torch.cat((jacobian1, jacobian2, inv_double_spacing_emb), dim=-1)
        jacobian = self.jacobian_calculator(jacobian)

        # Hessians
        # [BN, 2*edge_emb1+ 2*edge_emb1 + 2*edge_emb2] -> [BN, 2*node_emb]
        hessians = torch.cat(
            (half_jacobian_plus, half_jacobian_minus, inv_double_spacing_emb), dim=-1
        )
        hessians = self.hessian_calculator(hessians)

        # Result
        # [BN, 2*node_emb + 2*node_emb + 2*glob_emb + 2*node_emb] -> [BN, 2 * node_emb]
        result = torch.cat((node_emb, jacobian, glob_emb[batch], hessians), dim=-1)
        result = self.node_calculator(result)

        # Decoding: [BN, node_emb] -> [BN, 2]
        return node_attr * self.state_decoder(result)
