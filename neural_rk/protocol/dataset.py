from typing import Protocol

import pandas as pd
import torch_geometric.data as gData
from torch.utils.data.distributed import DistributedSampler

from neural_rk.dummy import DummySampler


class DatasetProtocol(Protocol):
    sampler: DistributedSampler | DummySampler

    def __init__(self, df: pd.DataFrame, window: int, take_only: float) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> gData.Data:
        ...
