import networkx as nx
import numpy as np
import numpy.typing as npt
from numba import njit

import graph.utils as gUtils

arr = npt.NDArray[np.float32]


@njit(fastmath=True)
def solve(graph: nx.Graph, weights: arr, temperature: arr, dts: arr) -> arr:
    """
    Solve heat equation exactly

    graph: underlying graph
    weights: [E, 1], dissipation rate of edges
    temperature: [N, 1], initial temperature of each node
    dts: [S, 1], dt for each time step


    weighted_laplacian_matrix: [N, N]
    temperature: (N, )
    dt: S-length list of (1,) ndarray

    return: (S+1, N)
    """
    weighted_laplacian_matrix = gUtils.get_weighted_laplacian_matrix(graph, weights)

    eig_val, eig_vec = np.linalg.eigh(weighted_laplacian_matrix)
    coeff = np.dot(temperature, eig_vec)

    trajectory = np.stack([np.empty_like(temperature)] * (len(dts) + 1))
    trajectory[0] = temperature

    t = np.array(0.0, dtype=np.float32)
    for step, dt in enumerate(dts):
        t += dt
        trajectory[step + 1] = np.sum(coeff * np.exp(-eig_val * t) * eig_vec, axis=1)
    return trajectory
