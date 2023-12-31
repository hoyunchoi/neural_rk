from __future__ import annotations

import torch
import torch.nn as nn

from neural_rk.protocol import ScalerProtocol
from neural_rk.scaler import IdentityScaler

from .mlp import MLP


class Encoder(nn.Module):
    def __init__(
        self,
        node_embedding_dims: list[int] = [],
        edge_embedding_dims: list[int] = [],
        glob_embedding_dims: list[int] = [],
        bn_momentum: float = 1.0,
        dropout: float = 0.0,
        scaler: ScalerProtocol = IdentityScaler(),
    ) -> None:
        """
        node_embedding_dims: list of node embedding dim, length is num node features
        edge_embedding_dims: list of edge embedding dim, length is num edge features
        glob_embedding_dims: list of glob embedding dim, length is num glob features
        bn_momentum: Refer get_batch_norm_layer
        """
        super().__init__()
        self.scaler = scaler

        self.node_encoder = nn.ModuleList(
            self.get_mlp_encoder(node_emb_dim, bn_momentum, dropout)
            for node_emb_dim in node_embedding_dims
        )

        self.edge_encoder = nn.ModuleList(
            self.get_mlp_encoder(edge_emb_dim, bn_momentum, dropout)
            for edge_emb_dim in edge_embedding_dims
        )

        self.glob_encoder = nn.ModuleList(
            self.get_mlp_encoder(glob_emb_dim, bn_momentum, dropout)
            for glob_emb_dim in glob_embedding_dims
        )

    @staticmethod
    def get_mlp_encoder(
        embedding_dim: int, bn_momentum: float, dropout: float
    ) -> nn.Module:
        if not embedding_dim:
            return nn.Identity()
        return MLP(1, embedding_dim, embedding_dim, bn_momentum, dropout)

    @staticmethod
    def encode(encoders: nn.ModuleList, tensors: torch.Tensor) -> torch.Tensor:
        """
        encoders: list of encoders, whose length is num_features. Each encoder: 1 -> embedding_dim
        tensors: list of tensors, whose length is num_feautres. Each tensor: [BN/BE/B, 1]
        """
        return torch.cat(
            [encoder(tensor) for encoder, tensor in zip(encoders, tensors)],
            dim=-1,
        )

    def forward(
        self,
        node_attr: torch.Tensor = torch.tensor([[]]),
        edge_attr: torch.Tensor = torch.tensor([[]]),
        glob_attr: torch.Tensor = torch.tensor([[]]),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        node_attr: [BN, len(node_embedding_dims)], node attributes that will be encoded
        edge_attr: [BE, len(edge_embedding_dims)], edge attributes that will be encoded
        glob_attr: [B, len(glob_embedding_dims), glob attributes that will be encoded

        Return
        node_atty: [BN, sum(node_embedding_dims)], embedding of node
        edge_attr: [BN, sum(edge_embedding_dims)], embedding of edge
        glob_attr: [B, sum(glob_embedding_dims)], embedding of glob
        """
        if node_attr.numel():
            node_attr = self.scaler(node_attr)
            node_emb = self.encode(self.node_encoder, node_attr.T.unsqueeze(-1))
        else:
            node_emb = node_attr

        if edge_attr.numel():
            edge_emb = self.encode(self.edge_encoder, edge_attr.T.unsqueeze(-1))
        else:
            edge_emb = edge_attr

        if glob_attr.numel():
            glob_emb = self.encode(self.glob_encoder, glob_attr.T.unsqueeze(-1))
        else:
            glob_emb = glob_attr

        return node_emb, edge_emb, glob_emb


class DuplicatedEncoder(nn.Module):
    """ Apply same MLP to each of the feautures """

    def __init__(
        self,
        node_embedding_dim: int = 0,
        edge_embedding_dim: int = 0,
        glob_embedding_dim: int = 0,
        bn_momentum: float = 1.0,
        dropout: float = 0.0,
        scaler: ScalerProtocol = IdentityScaler(),
    ) -> None:
        super().__init__()
        self.scaler = scaler

        self.node_encoder = self.get_mlp_encoder(
            node_embedding_dim, bn_momentum, dropout
        )
        self.edge_encoder = self.get_mlp_encoder(
            edge_embedding_dim, bn_momentum, dropout
        )
        self.glob_encoder = self.get_mlp_encoder(
            glob_embedding_dim, bn_momentum, dropout
        )

    @staticmethod
    def get_mlp_encoder(
        embedding_dim: int, bn_momentum: float, dropout: float
    ) -> nn.Module:
        if not embedding_dim:
            return nn.Identity()

        return MLP(1, embedding_dim, embedding_dim, bn_momentum, dropout)

    @staticmethod
    def encode(encoder: nn.Module, tensors: torch.Tensor) -> torch.Tensor:
        """
        encoder: A single encoder, which will be applied to tensors multiple times
        tensors: list of tensors, whose length is num_feautres. Each tensor: [BN/BE/B, 1]
        """
        return torch.cat([encoder(tensor) for tensor in tensors], dim=-1)

    def forward(
        self,
        node_attr: torch.Tensor = torch.tensor([[]]),
        edge_attr: torch.Tensor = torch.tensor([[]]),
        glob_attr: torch.Tensor = torch.tensor([[]]),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        node_attr: [BN, num_node_feature], node attributes that will be encoded
        edge_attr: [BE, num_edge_feature], edge attributes that will be encoded
        glob_attr: [B, num_glob_feature], glob attributes that will be encoded

        Return
        node_atty: [BN, num_node_feature*node_emb], embedding of node
        edge_attr: [BN, num_edge_feature*edge_emb], embedding of edge
        glob_attr: [B, num_glob_feature*glob_emb], embedding of glob
        """
        if node_attr.numel():
            node_attr = self.scaler(node_attr)
            node_emb = self.encode(self.node_encoder, node_attr.T.unsqueeze(-1))
        else:
            node_emb = node_attr

        if edge_attr.numel():
            edge_emb = self.encode(self.edge_encoder, edge_attr.T.unsqueeze(-1))
        else:
            edge_emb = edge_attr

        if glob_attr.numel():
            glob_emb = self.encode(self.glob_encoder, glob_attr.T.unsqueeze(-1))
        else:
            glob_emb = glob_attr

        return node_emb, edge_emb, glob_emb


class WholeEncoder(nn.Module):
    def __init__(
        self,
        node_embedding_dims: list[int] = [],
        edge_embedding_dims: list[int] = [],
        glob_embedding_dims: list[int] = [],
        bn_momentum: float = 1.0,
        dropout: float = 0.0,
        scaler: ScalerProtocol = IdentityScaler(),
    ) -> None:
        """
        node_embedding_dims: list of node embedding dim, length is num node features
        edge_embedding_dims: list of edge embedding dim, length is num edge features
        glob_embedding_dims: list of glob embedding dim, length is num glob features
        bn_momentum: Refer get_batch_norm_layer
        """

        super().__init__()
        self.scaler = scaler

        self.node_encoder = self.get_mlp_encoder(
            node_embedding_dims, bn_momentum, dropout
        )
        self.edge_encoder = self.get_mlp_encoder(
            edge_embedding_dims, bn_momentum, dropout
        )
        self.glob_encoder = self.get_mlp_encoder(
            glob_embedding_dims, bn_momentum, dropout
        )

    @staticmethod
    def get_mlp_encoder(
        embedding_dims: list[int], bn_momentum: float, dropout: float
    ) -> nn.Module:
        if not embedding_dims:
            return nn.Identity()

        feature_dim = len(embedding_dims)
        embedding_dim = sum(embedding_dims)
        return MLP(feature_dim, embedding_dim, embedding_dim, bn_momentum, dropout)

    def forward(
        self,
        node_attr: torch.Tensor = torch.tensor([[]]),
        edge_attr: torch.Tensor = torch.tensor([[]]),
        glob_attr: torch.Tensor = torch.tensor([[]]),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        node_attr: [BN, len(node_embedding_dims)], node attributes that will be encoded
        edge_attr: [BE, len(edge_embedding_dims)], edge attributes that will be encoded
        glob_attr: [B, len(glob_embedding_dims), glob attributes that will be encoded

        Return
        node_atty: [BN, sum(node_embedding_dims)], embedding of node
        edge_attr: [BN, sum(edge_embedding_dims)], embedding of edge
        glob_attr: [B, sum(glob_embedding_dims)], embedding of glob
        """
        if node_attr.numel():
            node_attr = self.scaler(node_attr)
            node_emb = self.node_encoder(node_attr)
        else:
            node_emb = node_attr

        if edge_attr.numel():
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = edge_attr

        if glob_attr.numel():
            glob_emb = self.glob_encoder(glob_attr)
        else:
            glob_emb = glob_attr

        return node_emb, edge_emb, glob_emb
