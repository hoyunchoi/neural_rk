import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch

sys.path.append(str(Path(__file__).parents[1]))

from neural_rk.path import DATA_DIR
from neural_rk.protocol import IsDivergingProtocol
from neural_rk.simulation_data import SimulationData, to_df
from graph import get_ba, get_er, get_rr
from graph.utils import get_degree_distribution, get_edge_list
from rossler import trajectory
from rossler.simulation import argument, solve


def main() -> None:
    is_diverging: IsDivergingProtocol = trajectory.IsDivergingPrecise()

    args = argument.get_args()
    rng = np.random.default_rng(args.seed)

    data: list[SimulationData] = []
    num_networks = 0
    while num_networks < args.num_networks:
        num_nodes = argument.get_num_nodes(args.num_nodes, rng)
        mean_degree = argument.get_mean_degree(args.mean_degree, rng)

        # * Graph-dependent variables
        network_type: Literal["er", "ba", "rr"] = rng.choice(args.network_type)
        if network_type == "ba":
            graph = get_ba(num_nodes, mean_degree, rng)
        elif network_type == "er":
            graph = get_er(num_nodes, mean_degree, rng=rng)
        else:
            graph = get_rr(num_nodes, mean_degree, rng=rng)

        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        edge_list = get_edge_list(graph)

        num_simulations, num_diverging = 0, 0
        while num_simulations < args.num_simulations and num_diverging < 5:
            params = argument.get_params(args.a, args.b, args.c, rng)  # [3, ]
            dts = argument.get_dt(args.steps, args.dt, rng)  # [S, 1]
            coupling = argument.get_coupling(  # [E, 1]
                num_edges, args.coupling, rng
            )

            # * Initial condition
            x = rng.uniform(-4.0, 4.0, size=num_nodes).astype(np.float32, copy=False)
            y = rng.uniform(-4.0, 4.0, size=num_nodes).astype(np.float32, copy=False)
            z = rng.uniform(0.0, 6.0, size=num_nodes).astype(np.float32, copy=False)
            positions = np.stack((x, y, z))  # [3, N]

            # * Solve Rossler equation
            trajectory = solve.solve(  # [S+1, 3, N]
                args.solver, graph, coupling, positions, dts, params
            )

            if is_diverging(torch.from_numpy(trajectory)):
                max_degree = max(get_degree_distribution(graph))
                print(f"{network_type=}, {mean_degree=}, {max_degree=}")
                num_diverging += 1
                continue

            # * Store the result
            num_simulations += 1
            data.append(
                {
                    "network_type": network_type,
                    "edge_index": edge_list,  # [E, 2]
                    "node_attr": np.empty((num_nodes, 0), dtype=np.float32),  # [N, 0]
                    "edge_attr": coupling,  # [E, 1]
                    "glob_attr": params[None, :],  # [1, 3]
                    "dts": dts,  # [S, 1]
                    "trajectories": trajectory.transpose(0, 2, 1),  # [S+1, N, 3]
                }
            )
        if num_simulations == args.num_simulations:
            num_networks += 1

    df = to_df(data)
    df.to_pickle(DATA_DIR / f"rossler_{args.name}.pkl")


if __name__ == "__main__":
    main()
