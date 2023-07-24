import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.append(str(Path(__file__).parents[1]))

from graph import get_ba, get_er, get_rr
from graph.utils import get_edge_list
from kuramoto import trajectory
from kuramoto.simulation import argument, solve
from neural_rk.path import DATA_DIR
from neural_rk.simulation_data import SimulationData, to_df


def main() -> None:
    args = argument.get_args()
    rng = np.random.default_rng(args.seed)
    rng_graph = (
        rng if args.seed_graph is None else np.random.default_rng(args.seed_graph)
    )
    is_diverging = trajectory.IsDivergingPrecise()

    # * Avoid possibly unbounded warning
    num_nodes, num_edges = 0, 0
    graph = nx.Graph()
    edge_list = np.array([], dtype=np.int64)
    dts = np.array([], dtype=np.float32)
    coupling = np.array([], dtype=np.float32)
    omega = np.array([], dtype=np.float32)
    initial_phase = np.array([], dtype=np.float32)

    # * Start simulation
    num_diverging = 0
    data: list[SimulationData] = []
    while len(data) < args.num_samples:
        if len(data) == 0 or not args.const_graph:
            # Graph setting
            network_type = argument.get_network_type(args.network_type, rng_graph)
            num_nodes = argument.get_num_nodes(args.num_nodes, rng_graph)
            mean_degree = argument.get_mean_degree(args.mean_degree, rng_graph)

            if network_type == "er":
                graph = get_er(num_nodes, mean_degree, rng=rng_graph)
            elif network_type == "ba":
                graph = get_ba(num_nodes, mean_degree, rng=rng_graph)
            else:
                graph = get_rr(num_nodes, mean_degree, rng=rng_graph)

            # Since only gcc is selected, the graph can have smaller num_nodes
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            edge_list = get_edge_list(graph)

        if len(data) == 0 or not args.const_dt:
            # dt setting
            try:
                dts = argument.get_dt(
                    args.max_time, args.steps, args.dt_delta, tuple(args.dt_clip), rng
                )
            except ValueError:
                print("Could not find proper dt")
                continue

        if len(data) == 0 or not args.const_param:
            # params(omega, coupling) setting
            omega = argument.get_omega(num_nodes, args.omega, rng)
            coupling = argument.get_coupling(num_edges, args.coupling, rng)

        if len(data) == 0 or not args.const_ic:
            # Initial condition setting
            initial_phase = argument.get_initial_condition(num_nodes, rng)

        # * Solve kuramoto equation
        phases = solve.solve(args.solver, graph, coupling, initial_phase, dts, omega)

        # * Check divergence of the trajectory
        if is_diverging(torch.from_numpy(phases)):
            # Divergence detected: drop the data
            num_diverging += 1
            print(f"{len(data)=}, {num_diverging=}")
            continue

        # * Store the result
        data.append(
            {
                "network_type": network_type,
                "edge_index": edge_list,  # [E, 2]
                "node_attr": omega,  # [N, 1]
                "edge_attr": coupling,  # [E, 1]
                "glob_attr": np.zeros((1, 0), dtype=np.float32),  # [1, 0]
                "dts": dts,  # [S, 1]
                "trajectories": phases,  # [S+1, N, 1]
            }
        )

    df = to_df(data)
    df.to_pickle(DATA_DIR / f"kuramoto_{args.name}.pkl")


if __name__ == "__main__":
    main()
