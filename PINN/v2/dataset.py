import numpy as np
import torch
from torch.utils.data import Dataset

tensor = torch.Tensor


class PINNDataset(Dataset):
    def __init__(
        self,
        xyt: tensor,
        field: tensor,
        sparsity: float,
        seed: int | None,
    ) -> None:
        """
        xyt: [num_data, 3]
        field: [num_data, 2]
        sparsity: Sparsity of choosing grid points for initial condition
        seed: random engine seed for randomly selecting data
        """
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        # Randomly prune data according to sparsity
        idx = self.rng.choice(len(xyt), int(sparsity * len(xyt)), replace=False)
        self.x, self.y, self.t = xyt[idx, :1], xyt[idx, 1:2], xyt[idx, 2:3]
        self.field = field[idx]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[tensor, tensor, tensor, tensor]:
        """Return x, y, t, uv"""
        return self.x[idx], self.y[idx], self.t[idx], self.field[idx]


class PeriodicBCDataset(Dataset):
    def __init__(self, xyt: tensor, sparsity: float, seed: int | None) -> None:
        """
        xyt: [Ny+1, Nx+1, S+1, 3]
        sparsity: Sparsity of choosing grid points for boundary condition
        """
        super().__init__()
        # bc1: left, bottom, bc2: right, top
        x1 = torch.cat((xyt[0, :, :, 0], xyt[1:, 0, :, 0]), dim=0).reshape(-1, 1)
        y1 = torch.cat((xyt[0, :, :, 1], xyt[1:, 0, :, 1]), dim=0).reshape(-1, 1)
        t1 = torch.cat((xyt[0, :, :, 2], xyt[1:, 0, :, 2]), dim=0).reshape(-1, 1)
        x2 = torch.cat((xyt[-1, :-1, :, 0], xyt[:, -1, :, 0]), dim=0).reshape(-1, 1)
        y2 = torch.cat((xyt[-1, :-1, :, 1], xyt[:, -1, :, 1]), dim=0).reshape(-1, 1)
        t2 = torch.cat((xyt[-1, :-1, :, 2], xyt[:, -1, :, 2]), dim=0).reshape(-1, 1)

        # Randomly prune data according to sparsity
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        idx = self.rng.choice(len(x1), int(sparsity * len(x1)), replace=False)
        self.x1, self.y1, self.t1 = x1[idx], y1[idx], t1[idx]
        self.x2, self.y2, self.t2 = x2[idx], y2[idx], t2[idx]

    def __len__(self) -> int:
        return len(self.x1)

    def __getitem__(
        self, idx: int
    ) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor]:
        """Return x1, y1, t1 for boundary1 and x2, y2, t2 for boundary2"""
        return (
            self.x1[idx],
            self.y1[idx],
            self.t1[idx],
            self.x2[idx],
            self.y2[idx],
            self.t2[idx],
        )
