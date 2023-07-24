import os
import time

import pandas as pd
import torch
import torch.multiprocessing as mp

from burgers.approximator import BurgersApproximator
from heat.approximator import HeatApproximator
from kuramoto.approximator import KuramotoApproximator
from neural_rk.experiment import run
from neural_rk.hyperparameter import HyperParameter, get_hp
from neural_rk.modules import prune_state_dict
from neural_rk.path import DATA_DIR, RESULT_DIR
from neural_rk.scaler import fit_in_scaler, fit_out_scaler
from rossler.approximator import RosslerApproximator


def main(hp: HyperParameter, save: bool = True) -> None:
    # ------------------------ Read data ------------------------
    start = time.perf_counter()

    train_df = pd.read_pickle(DATA_DIR / f"{hp.equation}_{hp.data}_train.pkl")
    val_df = pd.read_pickle(DATA_DIR / f"{hp.equation}_{hp.data}_val.pkl")

    # train_dataset = Dataset(train_df, hp.window, hp.take_only)
    # val_dataset = Dataset(val_df, hp.window, hp.take_only)
    # rollout_dataset = Dataset(val_df, window=-1, take_only=1.0)

    print(f"Reading data took {time.perf_counter()-start} seconds")

    # ------------------------ Create model ------------------------
    start = time.perf_counter()

    in_scaler = fit_in_scaler(hp.approximator, train_df.trajectories.tolist())
    out_scaler = fit_out_scaler(
        hp.approximator, train_df.trajectories.tolist(), train_df.dts.tolist()
    )

    if hp.equation == "heat":
        approximator = HeatApproximator.from_hp(
            hp.approximator, (in_scaler, out_scaler)
        )
    elif hp.equation == "rossler":
        approximator = RosslerApproximator.from_hp(
            hp.approximator, (in_scaler, out_scaler)
        )
    elif hp.equation == "kuramoto":
        approximator = KuramotoApproximator.from_hp(
            hp.approximator, (in_scaler, out_scaler)
        )
    elif hp.equation == "burgers":
        approximator = BurgersApproximator.from_hp(
            hp.approximator, (in_scaler, out_scaler)
        )
    else:
        raise ValueError(f"No such equation {hp.equation}")

    if hp.resume:
        checkpoint = torch.load(
            RESULT_DIR / f"{hp.equation}_{hp.resume}/best.pth",
            map_location=torch.device("cpu"),
        )
        approximator_state_dict = prune_state_dict(checkpoint["best_model_state_dict"])
        approximator.load_state_dict(approximator_state_dict)
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        grad_scaler_state_dict = checkpoint["grad_scaler_state_dict"]
    else:
        optimizer_state_dict = None
        grad_scaler_state_dict = None

    print(f"Creating model took {time.perf_counter()-start} seconds")

    # ------------------------ Training ------------------------
    start = time.perf_counter()

    if len(hp.device) == 1:
        run(
            0,
            hp,
            approximator,
            train_df,
            val_df,
            optimizer_state_dict,
            grad_scaler_state_dict,
            save,
        )
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = f"{hp.port}"

        mp.spawn(  # type:ignore
            run,
            args=(
                hp,
                approximator,
                train_df,
                val_df,
                optimizer_state_dict,
                grad_scaler_state_dict,
                save,
            ),
            nprocs=len(hp.device),
            join=True,
        )

    print(f"Training took {time.perf_counter()-start} seconds")


if __name__ == "__main__":
    hp = get_hp()
    main(hp)
