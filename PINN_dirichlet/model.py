import torch
import torch.nn as nn


def get_batch_norm_layer(in_channels: int, bn_momentum: float = -1.0) -> nn.Module:
    """Return batch normalization with given momentum
    if given momentum is 1.0, return identity layer"""
    if bn_momentum < 0.0:
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


def get_activation(activation: str) -> nn.Module:
    match:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU()
        case "elu":
            return nn.ELU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "leaky_relu":
            return nn.LeakyReLU()
        case _:
            raise ValueError(f"Invalid activation: {activation}")

def get_slp_layer(
    in_dim: int,
    out_dim: int,
    use_bias: bool,
    act: str = "gelu",
    bn_momentum: float,
    dropout: float,
    last: bool = False,
) -> list[nn.Module]:
    if last:
        return [nn.Linear(in_dim, out_dim, bias=use_bias)]
    else:
        return [
            nn.Linear(in_dim, out_dim, bias=use_bias),
            get_batch_norm_layer(out_dim, bn_momentum),
            get_activation(act),
            get_dropout_layer(dropout),
        ]


class Model(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 32,
        out_dim: int = 2,
        depth: int = 6,
        act: str = "gelu",
        bn_momentum: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        """
        Create 2-layer MLP module
        If last, do not use last activations
        """
        super().__init__()
        use_bias = bn_momentum == 1.0

        # 3 -> hidden -> ... -> hidden -> 2
        modules = get_slp_layer(in_dim, hidden_dim, use_bias, act, bn_momentum, dropout)
        for _ in range(depth - 2):
            modules.extend(
                get_slp_layer(hidden_dim, hidden_dim, use_bias, act, bn_momentum, dropout)
            )
        modules.extend(
            get_slp_layer(
                hidden_dim, out_dim, use_bias, act, bn_momentum, dropout, last=True
            )
        )

        self.mlp = nn.Sequential(*modules)

    def forward(self, *xyt: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat(xyt, dim=-1))
