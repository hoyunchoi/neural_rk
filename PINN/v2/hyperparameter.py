from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import yaml

from neural_rk.hyperparameter import EarlyStopParameter

if TYPE_CHECKING:
    from model import ACTIVATION


def extract_argument(kwargs: dict[str, Any], name: str) -> dict[str, Any]:
    keys = list(kwargs.keys())
    name += "_"
    extracted = {key.replace(name, ""): kwargs.pop(key) for key in keys if name in key}

    return extracted


@dataclass(slots=True)
class ModelParameter:
    hidden_dim: int
    depth: int
    bn_momentum: float
    dropout: float
    act: ACTIVATION

    @property
    def dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LossParameter:
    sparsity: float
    learning_rate: float
    optimizer: Literal["Adagrad", "Adam", "AdamW", "RMSprop", "SGD"]
    weight_decay: float
    batch_size: int
    seed: int | None


@dataclass(slots=True)
class HyperParameter:
    """Data class to save hyper parameters"""

    # * Data
    data_name: str
    data_idx: int

    # * NN and early_stop
    model: ModelParameter
    early_stop: EarlyStopParameter

    # * Loss: IC, BC, PDE, Data, Validation
    ic: LossParameter
    bc_value: LossParameter
    bc_derivative: LossParameter
    pde: LossParameter
    data: LossParameter
    val: LossParameter

    # * Train configuration
    epochs: int
    tqdm: bool
    device: torch.device

    @property
    def dict(self) -> dict[str, float]:
        return asdict(self)

    def to_yaml(self, file_path: Path | str) -> None:
        with open(file_path, "w") as f:
            yaml.safe_dump(self.dict, f)

    @classmethod
    def from_yaml(cls, file_path: Path | str) -> HyperParameter:
        with open(file_path, "r") as f:
            hp: dict[str, Any] = yaml.safe_load(f)
        model = ModelParameter(**hp.pop("model"))
        ic = LossParameter(**hp.pop("ic"))
        bc_value = LossParameter(**hp.pop("bc_value"))
        bc_derivative = LossParameter(**hp.pop("bc_derivative"))
        pde = LossParameter(**hp.pop("pde"))
        data = LossParameter(**hp.pop("data"))
        early_stop = EarlyStopParameter(**hp.pop("earlystop"))

        return cls(
            **hp, model=model, early_stop=early_stop, ic=ic, bc_value=bc_value, bc_derivative=bc_derivative, pde=pde, data=data
        )


def get_hp(options: list[str] | None = None) -> HyperParameter:
    parser = argparse.ArgumentParser()
    # * Data
    parser.add_argument("--data_name", default="IC_train")
    parser.add_argument("--data_idx", type=int, default=0)

    # * NN and early stop
    parser.add_argument("--model_hidden_dim", type=int, default=64)
    parser.add_argument("--model_depth", type=int, default=5)
    parser.add_argument("--model_bn_momentum", type=float, default=1.0)
    parser.add_argument("--model_dropout", type=float, default=0.0)
    parser.add_argument("--model_act", default="gelu")
    parser.add_argument(
        "--earlystop_patience",
        type=int,
        default=None,
        help="How many epochs to wait after validation loss is improved",
    )
    parser.add_argument(
        "--earlystop_delta",
        type=float,
        default=0.0,
        help="Minimum change of validation loss to regard as improved",
    )

    # * Loss: IC, BC, PDE, Data
    parser.add_argument("--ic_sparsity", type=float, default=1.0)
    parser.add_argument("--ic_learning_rate", type=float, default=1e-3)
    parser.add_argument("--ic_optimizer", default="AdamW")
    parser.add_argument("--ic_weight_decay", type=float, default=1e-4)
    parser.add_argument("--ic_batch_size", type=int, default=8192)
    parser.add_argument("--ic_seed", type=int, default=0)

    parser.add_argument("--bc_value_sparsity", type=float, default=0.1)
    parser.add_argument("--bc_value_learning_rate", type=float, default=1e-3)
    parser.add_argument("--bc_value_optimizer", default="AdamW")
    parser.add_argument("--bc_value_weight_decay", type=float, default=1e-4)
    parser.add_argument("--bc_value_batch_size", type=int, default=8192)
    parser.add_argument("--bc_value_seed", type=int, default=1)

    parser.add_argument("--bc_derivative_sparsity", type=float, default=0.1,help="Overwritten by bc value")
    parser.add_argument("--bc_derivative_learning_rate", type=float, default=1e-3)
    parser.add_argument("--bc_derivative_optimizer", default="AdamW")
    parser.add_argument("--bc_derivative_weight_decay", type=float, default=1e-4)
    parser.add_argument("--bc_derivative_batch_size", type=int, default=8192)
    parser.add_argument("--bc_derivative_seed", type=int, default=1, help="Overwritten by bc value")

    parser.add_argument("--pde_sparsity", type=float, default=0.01)
    parser.add_argument("--pde_learning_rate", type=float, default=1e-3)
    parser.add_argument("--pde_optimizer", default="AdamW")
    parser.add_argument("--pde_weight_decay", type=float, default=1e-4)
    parser.add_argument("--pde_batch_size", type=int, default=8192)
    parser.add_argument("--pde_seed", type=int, default=2)

    parser.add_argument("--data_sparsity", type=float, default=0.0, help="Overwritten by pde")
    parser.add_argument("--data_learning_rate", type=float, default=1e-3)
    parser.add_argument("--data_optimizer", default="AdamW")
    parser.add_argument("--data_weight_decay", type=float, default=1e-4)
    parser.add_argument("--data_batch_size", type=int, default=8192, help="Overwritten by pde")
    parser.add_argument("--data_seed", type=int, default=3, help="Overwritten by pde")

    parser.add_argument("--val_sparsity", type=float, default=0.01)
    parser.add_argument("--val_learning_rate", type=float, default=1e-3, help="Not used")
    parser.add_argument("--val_optimizer", default="AdamW", help="Not used")
    parser.add_argument("--val_weight_decay", type=float, default=1e-4, help="Not used")
    parser.add_argument("--val_batch_size", type=int, default=8192)
    parser.add_argument("--val_seed", type=int, default=3)

    # * Training
    parser.add_argument(
        "--epochs", type=int, default=300, help="Maximum number of epochs"
    )
    parser.add_argument("--tqdm", action="store_true", help="Use tqdm progress bar")
    parser.add_argument(
        "--device",
        default=["cuda:0"],
        choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="device to use",
        type=torch.device,
    )

    # Parse the arguments and return
    kwargs = vars(parser.parse_args(options))

    model_kwargs = extract_argument(kwargs, "model")
    earlystop_kwargs = extract_argument(kwargs, "earlystop")
    ic_kwargs = extract_argument(kwargs, "ic")
    bc_value_kwargs = extract_argument(kwargs, "bc_value")
    bc_derivative_kwargs = extract_argument(kwargs, "bc_derivative")
    pde_kwargs = extract_argument(kwargs, "pde")
    data_kwargs = extract_argument(kwargs, "data")

    return HyperParameter(
        model=ModelParameter(**model_kwargs),
        early_stop=EarlyStopParameter(**earlystop_kwargs),
        ic=LossParameter(**ic_kwargs),
        bc_value=LossParameter(**bc_value_kwargs),
        bc_derivative=LossParameter(**bc_derivative_kwargs),
        pde=LossParameter(**pde_kwargs),
        data=LossParameter(**data_kwargs),
        **kwargs,
    )
