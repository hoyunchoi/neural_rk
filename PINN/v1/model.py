import torch
import torch.nn as nn


def get_batch_norm_layer(in_channels: int, bn_momentum: float = 1.0) -> nn.Module:
    """Return batch normalization with given momentum
    if given momentum is 1.0, return identity layer"""
    if bn_momentum == 1.0:
        return nn.Identity()
    else:
        return nn.BatchNorm1d(in_channels, momentum=bn_momentum)


def get_dropout_layer(dropout: float = 0.0) -> nn.Module:
    """Return batch normalization with given momentum
    if given momentum is 1.0, return identity layer"""
    if dropout == 0.0:
        return nn.Identity()
    else:
        return nn.Dropout(dropout)


class Model(nn.Module):
    def __init__(
        self, hidden_dim: int = 32, depth: int = 6, bn_momentum: float = 1.0, dropout: float = 0.0
    ) -> None:
        """
        Create 2-layer MLP module
        If last, do not use last activations
        """
        super().__init__()
        use_bias = bn_momentum == 1.0

        def get_slp_layer(
            in_dim: int, out_dim: int, last: bool = False
        ) -> list[nn.Module]:
            if last:
                return [nn.Linear(in_dim, out_dim, bias=use_bias)]
            else:
                return [
                    nn.Linear(in_dim, hidden_dim, bias=use_bias),
                    get_batch_norm_layer(hidden_dim, bn_momentum),
                    nn.GELU(),
                    get_dropout_layer(dropout),
                ]

        # 3 -> hidden -> ... -> hidden -> 2
        modules = get_slp_layer(3, hidden_dim)
        for _ in range(depth - 2):
            modules.extend(get_slp_layer(hidden_dim, hidden_dim))
        modules.extend(get_slp_layer(hidden_dim, 2, last=True))

        self.mlp = nn.Sequential(*modules)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(torch.cat([x, y, t], dim=-1))
