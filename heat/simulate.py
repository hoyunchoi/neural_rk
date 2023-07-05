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
from heat import trajectory
from heat.simulation import argument, solve, solve_exact


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
            dts = argument.get_dt(args.steps, args.dt, rng)  # [S, 1]
            dissipation = argument.get_dissipation(  # [E, 1]
                num_edges, args.dissipation, rng
            )

            # * Initial condition
            temperature = np.zeros((num_nodes, 1), dtype=np.float32)  # [N, 1]
            random_hot_spots = rng.choice(
                num_nodes,
                size=rng.integers(num_nodes // 10, num_nodes // 10 * 9),
                replace=False,
                shuffle=False,
            )
            temperature[random_hot_spots, 0] = 1.0

            # * Solve heat equation
            trajectory = (
                solve_exact.solve(graph, dissipation, temperature, dts)
                if args.solver == "exact"
                else solve.solve(args.solver, graph, dissipation, temperature, dts)
            )

            if is_diverging(torch.from_numpy(trajectory)):
                dTs = np.abs(
                    [
                        sum(
                            (temperature[node] - temperature[i]).item()
                            for node in graph.neighbors(i)
                        )
                        for i in range(num_nodes)
                    ]
                )
                max_dT, max_node = np.amax(dTs), np.argmax(dTs)
                max_degree = max(get_degree_distribution(graph))
                print(
                    f"{network_type=}, {mean_degree=}, {max_dT=}, {max_node=},"
                    f" {max_degree=}"
                )
                num_diverging += 1
                continue

            # * Store the result
            num_simulations += 1
            data.append(
                {
                    "network_type": network_type,
                    "edge_index": edge_list,  # [E, 2]
                    "node_attr": np.empty((num_nodes, 0), dtype=np.float32),  # [N, 0]
                    "edge_attr": dissipation,  # [E, 1]
                    "glob_attr": np.empty((1, 0), dtype=np.float32),  # [1, 0]
                    "dts": dts,  # [S, 1]
                    "trajectories": trajectory,  # [S+1, N, 1]
                }
            )

        if num_simulations == args.num_simulations:
            num_networks += 1

    df = to_df(data)
    df.to_pickle(DATA_DIR / f"heat_{args.name}.pkl")


if __name__ == "__main__":
    main()
