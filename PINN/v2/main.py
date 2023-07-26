import copy
import string
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from dataset import PeriodicBCDataset, PINNDataset
from earlystop import EarlyStop
from hyperparameter import get_hp
from model import Model
from preprocess import preprocess
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from trainer import train_bc, train_ic, train_pde, validate

sys.path.append(str(Path(__file__).parents[2]))
from neural_rk.modules import count_trainable_param
from neural_rk.path import DATA_DIR, RESULT_DIR


def main() -> None:
    hp = get_hp()
    device = hp.device
    exp_name = "".join(
        np.random.choice(list(string.ascii_lowercase + string.digits), 8)
    )

    # Read dataframe and choose data to train/validate
    df = pd.read_pickle(DATA_DIR / f"burgers_{hp.data_name}.pkl")
    data = df.iloc[hp.data_idx]

    # Change to grid format
    # xyt: [Ny+1, Nx+1, S+1, 3], trajectory: [S+1, Ny+1, Nx+1, 2], nu: [2, ]
    xyt, trajectory, nu = preprocess(data)
    nu = nu.to(device)

    # Create dataset and corresponding dataloader
    dataset_ic = PINNDataset(
        xyt[:, :, 0].reshape(-1, 3),
        trajectory[0].reshape(-1, 2),
        hp.ic.sparsity,
        hp.ic.seed,
    )
    dataset_bc = PeriodicBCDataset(xyt, hp.bc.sparsity, hp.bc.seed)
    dataset_pde = PINNDataset(
        xyt.reshape(-1, 3), trajectory.reshape(-1, 2), hp.pde.sparsity, hp.pde.seed
    )
    dataset_val = PINNDataset(
        xyt.reshape(-1, 3), trajectory.reshape(-1, 2), hp.val.sparsity, hp.val.seed
    )
    dataloader_ic = DataLoader(
        dataset_ic,
        batch_size=hp.ic.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
    dataloader_bc = DataLoader(
        dataset_bc,
        batch_size=hp.bc.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
    dataloader_pde = DataLoader(
        dataset_pde,
        batch_size=hp.pde.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=hp.val.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
    )

    # Create model and optimizer
    model = Model(**hp.model.dict).to(device)
    optimizer_ic = getattr(optim, hp.ic.optimizer)(
        model.parameters(), lr=hp.ic.learning_rate, weight_decay=hp.ic.weight_decay
    )
    optimizer_bc_value = getattr(optim, hp.bc_value.optimizer)(
        model.parameters(),
        lr=hp.bc_value.learning_rate,
        weight_decay=hp.bc_value.weight_decay,
    )
    optimizer_bc_derivative = getattr(optim, hp.bc_derivative.optimizer)(
        model.parameters(),
        lr=hp.bc_derivative.learning_rate,
        weight_decay=hp.bc_derivative.weight_decay,
    )
    optimizer_pde = getattr(optim, hp.pde.optimizer)(
        model.parameters(), lr=hp.pde.learning_rate, weight_decay=hp.pde.weight_decay
    )
    optimizer_data = getattr(optim, hp.data.optimizer)(
        model.parameters(), lr=hp.data.learning_rate, weight_decay=hp.data.weight_decay
    )

    # Create early stop
    early_stop = EarlyStop.from_hp(hp.early_stop)

    print(f"Trainable parameters: {count_trainable_param(model)}")
    print(f"Number of initial condtion points: {len(dataset_ic)}")
    print(f"Number of boundary condtion points: {len(dataset_bc)}")
    print(f"Number of pde, data points: {len(dataset_pde)}")

    # Train
    losses: list[dict[str, float]] = []
    best_model_state_dict = copy.deepcopy(model.state_dict())

    epoch_range = trange(hp.epochs, file=sys.stdout) if hp.tqdm else range(hp.epochs)
    write: Callable[[str], None] = tqdm.write if hp.tqdm else print

    for epoch in epoch_range:
        # Training
        model.train()
        loss_ic = train_ic(model, dataloader_ic, optimizer_ic, device)
        loss_bc_value, loss_bc_derivative = train_bc(
            model, dataloader_bc, optimizer_bc_value, optimizer_bc_derivative, device
        )
        loss_pde, loss_data = train_pde(
            model, dataloader_pde, optimizer_pde, optimizer_data, nu, device
        )
        loss_train = loss_ic + loss_bc_value + loss_bc_derivative + loss_pde + loss_data

        # Validation
        model.eval()
        loss_val = validate(model, dataloader_val, device)

        # Store history
        losses.append(
            {
                "ic": loss_ic,
                "bc_value": loss_bc_value,
                "bc_derivative": loss_bc_derivative,
                "pde": loss_pde,
                "data": loss_data,
                "train": loss_train,
                "val": loss_val,
            }
        )

        # Early stop
        early_stop(loss_val)
        if early_stop.best_val:
            best_model_state_dict = copy.deepcopy(model.state_dict())
            write(f"{epoch}: train loss={loss_train:.4e}, val loss={loss_val:.4e}")
        if early_stop.abort:
            break

    # Store result
    result_dir = RESULT_DIR / exp_name
    result_dir.mkdir(parents=True, exist_ok=True)
    hp.to_yaml(result_dir / "hyperparameter.yaml")
    torch.save(best_model_state_dict, result_dir / "model.pt")
    pd.DataFrame.from_records(losses).to_csv(result_dir / "losses.pkl", sep="\t", index=False)


if __name__ == "__main__":
    main()
