import torch

from neural_rk.hyperparameter import ApproximatorParameter
from neural_rk.protocol import ScalerProtocol

from .identity import IdentityScaler
from .minmax import InverseMinMaxScaler, MinMaxScaler
from .sin_cos import SinCosScaler
from .standard import InverseStandardScaler, StandardScaler

STATE_DIM = {"heat": 1, "rossler": 3, "kuramoto": 1, "burgers": 2}


def get_scaler(equation_name: str, scaler_name: str, inverse: bool) -> ScalerProtocol:
    state_dim = STATE_DIM[equation_name]

    if scaler_name.lower() == "identity":
        return IdentityScaler(state_dim)
    elif scaler_name.lower() == "minmax":
        return InverseMinMaxScaler(state_dim) if inverse else MinMaxScaler(state_dim)
    elif scaler_name.lower() == "standard":
        return (
            InverseStandardScaler(state_dim) if inverse else StandardScaler(state_dim)
        )
    elif scaler_name.lower() == "sincos":
        return SinCosScaler(state_dim)
    raise ValueError(f"No such scaler: {scaler_name}")


def fit_in_scaler(
    hp: ApproximatorParameter, trajectories: list[torch.Tensor]
) -> ScalerProtocol:
    """
    trajectories: list of tensor of shape (S+1, N, state_features)
    """
    sampled_trajectories = torch.cat(  # ((S+1)N, state_features)
        [trajectory.reshape(-1, STATE_DIM[hp.equation]) for trajectory in trajectories]
    )

    in_scaler = get_scaler(hp.equation, hp.in_scaler, inverse=False)
    return in_scaler.fit(sampled_trajectories)


def fit_out_scaler(
    hp: ApproximatorParameter, trajectories: list[torch.Tensor], dts: list[torch.Tensor]
) -> ScalerProtocol:
    """
    trajectories: list of tensor of shape [S+1, N, state_features]
    dts: list of tensors of shape [S, 1]
    """

    def get_delta_traj(traj: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Delta trajectory / dt
        [S+1, N, state_features] -> [S*N, state_features]
        """
        return ((traj[1:] - traj[:-1]) / dt[..., None]).reshape(
            -1, STATE_DIM[hp.equation]
        )

    delta_trajectories = torch.cat(
        [get_delta_traj(traj, dt) for traj, dt in zip(trajectories, dts)]
    )

    out_scaler = get_scaler(hp.equation, hp.out_scaler, inverse=True)
    return out_scaler.fit(delta_trajectories)
