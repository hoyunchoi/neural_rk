import argparse

import numpy as np
import numpy.typing as npt


def get_args(options: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the dataset")

    # * Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If given, seed the random engine for both network and simulation",
    )

    # * Network parameters
    parser.add_argument(
        "--network_type",
        default=["er"],
        nargs="+",
        help="Type of networks",
        choices=["ba", "er", "rr", "ws"],
    )
    parser.add_argument(
        "--num_networks",
        type=int,
        default=10,
        help="Number of networks with different random seed",
    )
    parser.add_argument(
        "-N",
        "--num_nodes",
        type=int,
        nargs="+",
        default=[100],
        help="""Number of nodes at network.
        If one argument is given, it is constant.
        If two arguments are given, it is determined to be random value between two numbers (inclusive).""",
    )
    parser.add_argument(
        "-M",
        "--mean_degree",
        type=float,
        nargs="+",
        default=[4.0],
        help="""Mean degree of network.
        If one argument is given, it is constant.
        If two arguments are given, it is determined to be random value between two numbers (inclusive).""",
    )

    # * Simulation parameters
    parser.add_argument(
        "--solver",
        choices=["rk1", "rk2", "rk4"],
        default="rk4",
        help="How to solve kuramoto equation",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=1,
        help="Number of simulations per network with different initial conditions",
    )
    parser.add_argument(
        "--steps", type=int, default=800, help="Number of time steps per simulation"
    )
    parser.add_argument(
        "--dt",
        type=float,
        nargs="+",
        default=[0.01],
        help="""dt for Runge-Kutta method.
        If one argument is given, it is const for every time step.
        If two arguments are given, it is determined to be random value between two numbers (inclusive).""",
    )
    parser.add_argument(
        "-C",
        "--coupling",
        type=float,
        nargs="+",
        default=[0.01],
        help="""Coupling constant.
        If one argument is given, it is constant for every edge.
        If two arguments are given, it is determined to be random value between two numbers (inclusive).""",
    )
    parser.add_argument(
        "--omega",
        nargs=3,
        default=["uniform", "1.0", "1.0"],
        help="""Distribution of omega.
        If normal distribution, second argument is mean and third argumet is std
        If uniform distribution, second argument is min and third argument is max (inclusive).""",
    )

    # Parse the arguments and return
    if options is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args=options)


def get_num_nodes(args_num_nodes: list[int], rng: np.random.Generator) -> int:
    return rng.integers(min(args_num_nodes), max(args_num_nodes))


def get_mean_degree(args_mean_degree: list[float], rng: np.random.Generator) -> float:
    return rng.uniform(min(args_mean_degree), max(args_mean_degree))


def get_omega(
    num_nodes: int, args_omega: list[str], rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """return [N, 1]"""
    if args_omega[0] == "normal":
        avg, std = float(args_omega[1]), float(args_omega[2])
        return rng.normal(avg, std, size=(num_nodes, 1)).astype(np.float32)

    elif args_omega[0] == "uniform":
        low, high = float(args_omega[1]), float(args_omega[2])
        return rng.uniform(low, high, size=(num_nodes, 1)).astype(np.float32)

    raise ValueError(f"No such omega distribution {args_omega[0]}")


def get_dt(
    steps: int, args_dt: list[float], rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """return: [S, 1]"""
    return rng.uniform(min(args_dt), max(args_dt), size=(steps, 1)).astype(np.float32)


def get_coupling(
    num_edges: int, args_coupling: list[float], rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """return: (E, 1)"""
    return rng.uniform(
        min(args_coupling), max(args_coupling), size=(num_edges, 1)
    ).astype(np.float32)
