from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

tensor = torch.Tensor


class ICDataset(Dataset):
    def __init__(
        self, xyt: tensor, field: tensor, sparsity: float, rng: np.random.Generator
    ) -> None:
        """
        xyt: [Ny+1, Nx+1, S+1, 3]
        field: [S+1, Ny+1, Nx+1, 2]
        sparsity: Sparsity of choosing grid points for initial condition
        """
        super().__init__()
        x = xyt[:, :, 0, 0].reshape(-1, 1)
        y = xyt[:, :, 0, 1].reshape(-1, 1)
        t = torch.zeros_like(x)
        uv = field[0].reshape(-1, 2)

        idx = rng.choice(len(x), int(sparsity * len(x)), replace=False)
        self.x, self.y, self.t, self.uv = x[idx], y[idx], t[idx], uv[idx]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[tensor, tensor, tensor, tensor]:
        """Return x, y, t, uv"""
        return self.x[idx], self.y[idx], self.t[idx], self.uv[idx]


class PeriodicBCDataset(Dataset):
    def __init__(self, xyt: tensor, sparsity: float, rng: np.random.Generator) -> None:
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
        # requires_grad = True for derivative of boundary
        idx = rng.choice(len(x1), int(sparsity * len(x1)), replace=False)
        self.x1 = x1[idx].requires_grad_(True)
        self.y1 = y1[idx].requires_grad_(True)
        self.t1 = t1[idx].requires_grad_(True)
        self.x2 = x2[idx].requires_grad_(True)
        self.y2 = y2[idx].requires_grad_(True)
        self.t2 = t2[idx].requires_grad_(True)

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


class PDEDataset(Dataset):
    def __init__(self, xyt: tensor, sparsity: float, rng: np.random.Generator) -> None:
        """
        xyt: [Ny+1, Nx+1, S+1, 3]
        sparsity: Sparsity of choosing grid points for pde condition
        """
        super().__init__()

        x = xyt[..., 0].reshape(-1, 1)
        y = xyt[..., 1].reshape(-1, 1)
        t = xyt[..., 2].reshape(-1, 1)

        # Randomly prune data according to sparsity
        # requires_grad = True for derivative
        idx = rng.choice(len(x), int(sparsity * len(x)), replace=False)
        self.x = x[idx].requires_grad_(True)
        self.y = y[idx].requires_grad_(True)
        self.t = t[idx].requires_grad_(True)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[tensor, tensor, tensor]:
        return self.x[idx], self.y[idx], self.t[idx]


class DataDataset(Dataset):
    def __init__(
        self, xyt: tensor, field: tensor, sparsity: float, rng: np.random.Generator
    ) -> None:
        """
        xyt: [Ny+1, Nx+1, S+1, 3]
        sparsity: Sparsity of choosing grid points for data condition
        field: [S+1, Ny+1, Nx+1, 2]
        """
        x = xyt[..., 0].reshape(-1, 1)
        y = xyt[..., 1].reshape(-1, 1)
        t = xyt[..., 2].reshape(-1, 1)
        uv = field.reshape(-1, 2)

        # Randomly prune data according to sparsity
        idx = rng.choice(len(x), int(sparsity * len(x)), replace=False)
        self.x, self.y, self.t, self.uv = x[idx], y[idx], t[idx], uv[idx]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[tensor, tensor, tensor, tensor]:
        return self.x[idx], self.y[idx], self.t[idx], self.uv[idx]


@dataclass
class PINNDataset:
    ic: ICDataset
    bc: PeriodicBCDataset
    pde: PDEDataset
    data: DataDataset
