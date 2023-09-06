import sys
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch

sys.path.append(str(Path(__file__).parents[1]))

from burgers import trajectory
from burgers.simulation import argument, solve_grid
from graph import get_square_grid
from graph.utils import get_edge_list
from neural_rk.grid_graph import dxdy2edge, dxdy2pos
from neural_rk.path import DATA_DIR
from neural_rk.simulation_data import SimulationData, to_df

arr = npt.NDArray[np.float32]
Lx, Ly, max_time = 1.0, 1.0, 1.0  # x, y, t length of the domain


def main() -> None:
    args = argument.get_args()
    rng = np.random.default_rng(args.seed)
    rng_ic = rng if args.seed_ic is None else np.random.default_rng(args.seed_ic)
    is_diverging = trajectory.IsDiverging()

    # Avoid possibly unbounded warning
    Nx, Ny, ndim = 0, 0, 0
    dx = np.array([], dtype=np.float32)
    dy = np.array([], dtype=np.float32)
    position = np.array([], dtype=np.float32)
    nu = (0.0, 0.0)
    dts = np.array([], dtype=np.float32)
    get_initial_condition: Callable[[arr], arr] = lambda x: x
    edge_list = np.array([], dtype=np.int64)
    edge_attr = np.array([], dtype=np.float32)
    node_attr = np.array([], dtype=np.float32)

    # start simulation
    num_diverging = 0
    data: list[SimulationData] = []
    while len(data) < args.num_samples:
        if len(data) == 0 or not args.const_graph:
            # Grid setting
            Nx, Ny = argument.get_NxNy(args.Nx, args.Ny, rng)
            ndim = int(Nx > 1) + int(Ny > 1)  # 1 or 2

            try:
                dx, dy = argument.get_dxdy(
                    (Lx, Ly), (Nx, Ny), args.dx_delta, tuple(args.dx_clip), rng
                )
            except ValueError:
                print("Could not find proper dx, dy")
                continue
            position = dxdy2pos(dx, dy)  # [Ny+1, Nx+1, 2]

            # Graph setting
            graph = get_square_grid(Nx, Ny)
            edge_list = get_edge_list(graph)
            edge_attr = dxdy2edge(dx, dy)
            node_attr = np.ones((graph.number_of_nodes(), 2), dtype=np.float32)
            if ndim == 1:
                node_attr[..., 1] = 0

        if len(data) == 0 or not args.const_dt:
            # dt setting
            try:
                dts = argument.get_dt(
                    max_time, args.steps, args.dt_delta, tuple(args.dt_clip), rng
                )
            except ValueError:
                print("Could not find proper dt")
                continue

        if len(data) == 0 or not args.const_param:
            # params(nu) setting
            nu = argument.get_nu(ndim, args.nu, rng)

        if len(data) == 0 or not args.const_ic:
            # Initial condition setting
            get_initial_condition = argument.get_initial_condition(
                (Lx, Ly),
                ndim,
                args.phase,
                args.offset,
                rng_ic,
            )

        # Get initial field
        # Due to periodic b.c., last position will be ignored
        initial_field = get_initial_condition(position[:-1, :-1])  # [Ny, Nx, 2]

        # Solve burgers equation
        fields = solve_grid.solve(  # [S+1, Nx*Ny, 2]
            args.solver, (dx, dy), nu, initial_field, dts
        )

        # Check divergence of the trajectory
        if is_diverging(torch.from_numpy(fields)):
            # Divergence detected: drop the data
            num_diverging += 1
            print(f"{len(data)=}, {num_diverging=}")
            continue

        # Store the result
        data.append(
            {
                "network_type": f"{Nx}_{Ny}",
                "edge_index": edge_list,  # [E, 2]
                "node_attr": node_attr,  # [N, 2]
                "edge_attr": edge_attr,  # [E, 2]
                "glob_attr": np.array([nu], dtype=np.float32),  # [1, 2]
                "dts": dts,  # [S, 1]
                "trajectories": fields,  # [S+1, N, 2]
            }
        )

    df = to_df(data, directed=True)
    df.to_pickle(DATA_DIR / f"burgers_{args.name}.pkl")


if __name__ == "__main__":
    np.seterr(over="raise")
    main()
