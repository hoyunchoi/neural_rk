from __future__ import annotations

import argparse
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Type

import yaml


def extract_argument(args: dict[str, Any], name: str) -> dict[str, Any]:
    keys = list(args.keys())
    name += "_"
    extracted = {key.replace(name, ""): args.pop(key) for key in keys if name in key}

    return extracted


def get_hp(options: list[str] | None = None) -> HyperParameter:
    """
    Save argument values to hyper parameter instance
    Args
        options: arguments from jupyter kernel
    """
    parser = argparse.ArgumentParser()

    # * Data
    parser.add_argument(
        "--equation",
        default="rossler",
        choices=["heat", "rossler", "kuramoto", "burgers"],
        help="Equation name",
    )
    parser.add_argument(
        "--data",
        default="1",
        help="data file name without parent directories and extension (.pkl)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Noise level to corrupt input data while training",
    )
    parser.add_argument(
        "--take_only",
        type=float,
        default=1.0,
        help="If less than 1, only use portion of data for training/validation.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="How many time step will model predict in train step",
    )

    # * NN
    parser.add_argument(
        "--rk", default="RK1", choices=["RK1", "RK2", "RK4"], help="GNN model name"
    )
    parser.add_argument(
        "--approximator_in_scaler",
        default="identity",
        choices=["identity", "standard", "minmax", "sincos"],
        help="Scaler to transform input of modĉl",
    )
    parser.add_argument(
        "--approximator_out_scaler",
        default="identity",
        choices=["identity", "standard", "minmax"],
        help="Scaler to transform output of model",
    )
    parser.add_argument(
        "--approximator_state_embedding",
        nargs="+",
        type=int,
        default=[],
        help="Embedding dimensions for each states attributes",
    )
    parser.add_argument(
        "--approximator_node_embedding",
        nargs="+",
        type=int,
        default=[],
        help="Embedding dimensions for each node attributes",
    )
    parser.add_argument(
        "--approximator_edge_embedding",
        nargs="+",
        type=int,
        default=[],
        help="Embedding dimensions for each edge attributes",
    )
    parser.add_argument(
        "--approximator_glob_embedding",
        nargs="+",
        type=int,
        default=[],
        help="Embedding dimensions for each global attributes",
    )
    parser.add_argument(
        "--approximator_edge_hidden",
        type=int,
        default=64,
        help="Dimension of hidden layer for mlp at approximating interaction",
    )
    parser.add_argument(
        "--approximator_node_hidden",
        type=int,
        default=64,
        help="Dimension of hidden layer for mlp at aggregating interaction",
    )
    parser.add_argument(
        "--approximator_dropout", type=float, default=0.0, help="dropout rate"
    )
    parser.add_argument(
        "--approximator_bn_momentum",
        type=float,
        default=1.0,
        help="Batch normalization momentum. If given, apply batch normalization",
    )

    # * Loss and Optimizer
    parser.add_argument(
        "--loss",
        default="MSE",
        choices=["MSE", "MAE", "MSE_heat", "MSE_kuramoto"],
        help="Metric name to use as loss for back-propagation",
    )
    parser.add_argument(
        "--optimizer",
        default="AdamW",
        choices=["Adagrad", "Adam", "AdamW", "RMSprop", "SGD"],
        help="torch optimizer names",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay for optimizer"
    )

    # * scheduler
    parser.add_argument(
        "--scheduler_name",
        default="cosine",
        choices=["cosine", "step", "exponential"],
        help="Which scheduler to control learning rate",
    )
    parser.add_argument(
        "--scheduler_lr",
        type=float,
        default=1e-4,
        help="learning rate. This will be the minimum learning rate for scheduler",
    )
    parser.add_argument(
        "--scheduler_lr_max",
        type=float,
        default=1e-3,
        help="Maximum learning rate.",
    )
    parser.add_argument(
        "--scheduler_period",
        type=int,
        default=20,
        help="Number of epochs for scheduler cycle/",
    )
    parser.add_argument(
        "--scheduler_warmup",
        type=int,
        default=0,
        help="Number of epochs for warm up stage. If 0, skip warmup stage",
    )
    parser.add_argument(
        "--scheduler_lr_max_mult",
        type=float,
        default=0.5,
        help="Decreasing rate of lr_max per each cycle",
    )
    parser.add_argument(
        "--scheduler_period_mult",
        type=float,
        default=1.5,
        help="Multiplier of period of scheduler.",
    )

    # * Early Stop
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

    # * Train configuration
    parser.add_argument("--resume", help="experiment id to resume training")
    parser.add_argument(
        "--device",
        nargs="+",
        default=["0"],
        choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "0", "1", "2", "3"],
        help="device to use",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3184,
        help="Localhost port for multiprocessing commnication",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for torch random")
    parser.add_argument(
        "--epochs", type=int, default=162, help="Maximum number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=None,
        help=(
            "batch size of validation rollout. If not given, use batch size as default"
        ),
    )
    parser.add_argument(
        "--tqdm", action="store_true", help="If this flag is on, use tqdm"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="If this flag is on, use wandb"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="If this flag is on, use amp (automatic mixed precision)",
    )

    # Parse the arguments and return
    args = vars(parser.parse_args(options))

    approximator_args = extract_argument(args, "approximator")
    approximator = ApproximatorParameter(equation=args["equation"], **approximator_args)

    scheduler_args = extract_argument(args, "scheduler")
    scheduler = SchedulerParameter(**scheduler_args)

    earlystop_args = extract_argument(args, "earlystop")
    earlystop = EarlyStopParameter(**earlystop_args)

    return HyperParameter(
        approximator=approximator,
        scheduler=scheduler,
        earlystop=earlystop,
        **args,
    )


@dataclass(slots=True)
class EarlyStopParameter:
    patience: int | None
    delta: float


@dataclass(slots=True)
class SchedulerParameter:
    name: Literal["cosine", "step", "exponential"]
    lr: float
    lr_max: float
    period: int
    warmup: int
    lr_max_mult: float
    period_mult: float

    def __post_init__(self) -> None:
        # scheduler dependent variables
        assert self.lr <= self.lr_max
        if self.name == "step":
            self.warmup = 0
        elif self.name == "exponential":
            self.warmup = 0
            self.period = 1
            self.period_mult = 1.0


@dataclass(slots=True)
class ApproximatorParameter:
    equation: Literal["heat", "rossler", "kuramoto", "burgers"]
    in_scaler: Literal["identity", "standard", "minmax", "sincos"]
    out_scaler: Literal["identity", "standard", "minmax"]
    state_embedding: list[int]
    node_embedding: list[int]
    edge_embedding: list[int]
    glob_embedding: list[int]
    edge_hidden: int
    node_hidden: int
    dropout: float
    bn_momentum: float

    def __post_init__(self) -> None:
        # Equation dependent variables
        if self.equation == "heat":
            assert len(self.state_embedding) == 1
            assert len(self.node_embedding) == 0
            assert len(self.edge_embedding) == 1
            assert len(self.glob_embedding) == 0
        elif self.equation == "rossler":
            assert len(self.state_embedding) == 3
            assert len(self.node_embedding) == 0
            assert len(self.edge_embedding) == 1
            assert len(self.glob_embedding) == 3
        elif self.equation == "kuramoto":
            assert (
                self.in_scaler == "sincos"
            ), "For kuramoto, you should use sincos scaler"
            assert len(self.state_embedding) == 1
            assert len(self.node_embedding) == 1
            assert len(self.edge_embedding) == 1
            assert len(self.glob_embedding) == 0
        elif self.equation == "burgers":
            assert len(self.state_embedding) == 1
            assert len(self.node_embedding) == 0
            assert len(self.edge_embedding) <= 2
            assert len(self.glob_embedding) == 1
            if len(self.edge_embedding) == 1:
                self.edge_embedding *= 2
        else:
            raise NotImplementedError(f"No such equation: {self.equation}")


@dataclass(slots=True)
class HyperParameter:
    """Data class to save hyper parameters"""

    # * Data
    equation: Literal["heat", "rossler", "kuramoto", "burgers"]
    data: str
    noise: float
    take_only: float
    window: int

    # * NN
    rk: Literal["RK1", "RK2", "RK4"]
    approximator: ApproximatorParameter

    # * Loss and Optimizer
    loss: Literal["MSE", "MAE", "MSE_heat", "MSE_kuramoto"]
    optimizer: Literal["Adagrad", "Adam", "AdamW", "RMSprop", "SGD"]
    weight_decay: float

    # * scheduler and earely stop
    scheduler: SchedulerParameter
    earlystop: EarlyStopParameter

    # * Train configuration
    resume: str | None
    device: list[str]
    port: int
    seed: int | None
    epochs: int
    batch_size: int
    rollout_batch_size: int
    tqdm: bool
    wandb: bool
    amp: bool

    def __post_init__(self) -> None:
        # Train configuration
        if self.rollout_batch_size is None:
            self.rollout_batch_size = self.batch_size

        # Loss configuration
        if self.equation != "heat":
            assert "heat" not in self.loss
        elif self.equation != "kuramoto":
            assert "kuramoto" not in self.loss

        # Device configuration
        self.device = [
            f"cuda:{device}" if len(device) == 1 else device for device in self.device
        ]

        # Seed warning
        if self.seed is not None:
            warnings.warn(
                "The neural network uses scatter sum, which is inherently random. The"
                " seed will be ignored"
            )

    def to_yaml(self, file_path: Path | str) -> None:
        with open(file_path, "w") as f:
            yaml.safe_dump(self.dict, f)

    @classmethod
    def from_yaml(cls, file_path: Path | str) -> HyperParameter:
        with open(file_path, "r") as f:
            hp: dict[str, Any] = yaml.safe_load(f)
        approximator = ApproximatorParameter(**hp.pop("approximator"))
        scheduler = SchedulerParameter(**hp.pop("scheduler"))
        earlystop = EarlyStopParameter(**hp.pop("earlystop"))

        return cls(
            **hp, approximator=approximator, scheduler=scheduler, earlystop=earlystop
        )

    @property
    def dict(self) -> dict[str, Any]:
        return asdict(self)

    def get_type(self, member_name: str) -> Type:
        return type(getattr(self, member_name))


if __name__ == "__main__":
    # hp = get_hp()
    # print(hp)
    # hp.to_yaml("temp.yaml")

    hp2 = HyperParameter.from_yaml("temp.yaml")
    print(hp2)
    # print(hp2.scheduler)
