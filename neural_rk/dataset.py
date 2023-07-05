from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.utils.data as tData
import torch_geometric.data as gData

from .dummy import DummySampler
from .protocol import DatasetProtocol


class Dataset(tData.Dataset):
    def __init__(
        self, df: pd.DataFrame, window: int = 1, take_only: float = 1.0, seed: int = 42
    ) -> None:
        """
        df: Dataframe to extract gData from
        window: How many time step will model predict
        take_only: If less than 1, store trajectory with only given probability

        N: number of nodes
        E: number of edges
        S: number of steps
        W: window
        """
        assert window != 0, f"Window should be nonzero"
        self.sampler: tData.DistributedSampler | DummySampler = DummySampler()

        # Number of total steps each sample has, considering time window
        steps_per_sample = [len(trajectory) for trajectory in df.trajectories]
        if window < 0:
            assert all_equal(steps_per_sample), (
                "If you want to do full rollout with batch, "
                "number of steps of each samples should be equal"
            )
            window += steps_per_sample[0]
        steps_per_sample = [num_step - window for num_step in steps_per_sample]

        # Randomly sample data from entire steps
        if take_only == 1.0:
            chosen_step_per_sample = [
                np.arange(num_step) for num_step in steps_per_sample
            ]
        else:
            random_engine = np.random.default_rng(seed=seed)
            chosen_step_per_sample = [
                random_choice(random_engine, num_step, take_only)
                for num_step in steps_per_sample
            ]

        # store sampled data
        self.data: list[gData.Data] = []
        for chosen_step, (_, series) in zip(chosen_step_per_sample, df.iterrows()):
            self.data.extend(
                [
                    gData.Data(
                        x=series.trajectories[step],  # (N, state_dim)
                        edge_index=series.edge_index,  # (2, 2E)
                        dt=series.dts[step : step + window].unsqueeze(0),  # (1, W, 1)
                        node_attr=series.node_attr,  # (N, node_dim)
                        edge_attr=series.edge_attr,  # (2E, edge_dim)
                        glob_attr=series.glob_attr,  # (1, glob_dim)
                        y=torch.divide(  # (N, W, state_dim)
                            series.trajectories[step + 1 : step + window + 1]
                            - series.trajectories[step : step + window],
                            series.dts[step : step + window, None],
                        ).transpose(0, 1),
                    )
                    for step in chosen_step
                ]
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> gData.Data:
        return self.data[index]


class EfficientDataset(tData.Dataset):
    def __init__(
        self, df: pd.DataFrame, window: int = 1, take_only: float = 1.0, seed: int = 42
    ) -> None:
        """
        Memory-efficient dataset, but slow when iterate over dataloader
        take_only: If less than 1, store trajectory with only given probability

        N: number of nodes
        E: number of edges
        S: number of steps
        """
        self.df = df
        self.sampler: tData.DistributedSampler | DummySampler = DummySampler()

        # Number of total steps each sample has, considering time window
        steps_per_sample = [len(trajectory) for trajectory in df.trajectories]
        if window < 0:
            assert all_equal(steps_per_sample), (
                "If you want to do full rollout with batch, "
                "number of steps of each samples should be equal"
            )
            window += steps_per_sample[0]
        self.window = window
        steps_per_sample = [num_step - self.window for num_step in steps_per_sample]

        # Randomly sample data from entire steps
        self.chosen_step_per_sample: list[npt.NDArray[np.int64]]
        if take_only == 1.0:
            self.chosen_step_per_sample = [
                np.arange(num_step) for num_step in steps_per_sample
            ]
        else:
            random_engine = np.random.default_rng(seed=seed)
            self.chosen_step_per_sample = [
                random_choice(random_engine, num_step, take_only)
                for num_step in steps_per_sample
            ]

        # (Sample index, Step) for each __getitem__ index
        self._ptr: list[tuple[int, int]] = []
        for sample_idx, chosen_step in enumerate(self.chosen_step_per_sample):
            self._ptr.extend((sample_idx, step) for step in chosen_step)

    def __len__(self) -> int:
        return len(self._ptr)

    def __getitem__(self, index: int) -> gData.Data:
        sample_idx, step = self._ptr[index]

        series = self.df.iloc[sample_idx]
        return gData.Data(
            x=series.trajectories[step],  # (N, state_dim)
            edge_index=series.edge_index,  # (2, 2E)
            dt=series.dts[step : step + self.window].unsqueeze(0),  # (1, W, 1)
            node_attr=series.node_attr,  # (N, node_dim)
            edge_attr=series.edge_attr,  # (2E, edge_dim)
            glob_attr=series.glob_attr,  # (1, glob_dim)
            y=torch.divide(  # (N, W, state_dim)
                series.trajectories[step + 1 : step + self.window + 1]
                - series.trajectories[step : step + self.window],
                series.dts[step : step + self.window].unsqueeze(-1),
            ).transpose(0, 1),
        )


def random_choice(
    random_engine: np.random.Generator, steps_per_sample: int, take_only: float
) -> npt.NDArray[np.int64]:
    randomly_chosen = random_engine.choice(
        steps_per_sample,
        max(1, int(take_only * steps_per_sample)),
        replace=False,
        shuffle=False,
    )
    return np.sort(randomly_chosen)


def all_equal(numbers: list[int]) -> bool:
    return all(numbers[0] == x for x in numbers)


def collate_fn(data: list[gData.data.BaseData]):
    batch_data = gData.Batch.from_data_list(data)

    # (BN, W, state_feature) -> (W, BN, stae_feature) -> (BN, state_feature) if W=1
    batch_data.y = torch.transpose(batch_data.y, 0, 1).squeeze(0)

    # (B, W, 1) -> (BN, W, 1) -> (W, BN, 1) -> (BN, 1) if W = 1
    batch_data.dt = batch_data.dt[batch_data.batch]
    batch_data.dt = torch.transpose(batch_data.dt, 0, 1).squeeze(0)
    return batch_data


def get_data_loader(
    dataset: DatasetProtocol, is_ddp: bool, device: torch.device | str = "", **kwargs
) -> tData.DataLoader:
    device = str(device) if kwargs.get("pin_memory", False) else ""
    shuffle = kwargs.pop("shuffle")

    if is_ddp:
        dataset.sampler = tData.DistributedSampler(
            cast(tData.Dataset, dataset), shuffle=shuffle
        )
        return tData.DataLoader(
            cast(tData.Dataset, dataset),
            sampler=dataset.sampler,
            pin_memory_device=device,
            collate_fn=collate_fn,
            **kwargs,
        )
    else:
        return tData.DataLoader(
            cast(tData.Dataset, dataset),
            shuffle=shuffle,
            pin_memory_device=device,
            collate_fn=collate_fn,
            **kwargs,
        )
