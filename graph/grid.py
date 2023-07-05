import itertools

import networkx as nx


def get_1d_square_grid(Nx: int) -> nx.DiGraph:
    grid = nx.DiGraph()
    for node in range(Nx):
        grid.add_edge(node, (node + 1) % Nx, axis="x")

    # Remove self-loop
    grid.remove_edges_from(nx.function.selfloop_edges(grid))
    return grid


def get_2d_square_grid(Nx: int, Ny: int) -> nx.DiGraph:
    grid = nx.DiGraph()
    for x, y in itertools.product(range(Nx), range(Ny)):
        grid.add_edge((x, y), ((x + 1) % Nx, y), axis="x")
        grid.add_edge((x, y), (x, (y + 1) % Ny), axis="y")

    # Relabel 2D position to corresponding indices
    pos2idx = {
        (x, y): i for i, (y, x) in enumerate(itertools.product(range(Ny), range(Nx)))
    }
    grid = nx.relabel.relabel_nodes(grid, pos2idx)

    # Sort node idx
    graph = nx.DiGraph()
    graph.add_nodes_from(range(grid.number_of_nodes()))
    graph.add_edges_from(grid.edges)

    # Remove self-loop
    graph.remove_edges_from(nx.function.selfloop_edges(graph))
    return graph


def get_3d_square_grid(Nx: int, Ny: int, Nz: int) -> nx.DiGraph:
    grid = nx.DiGraph()
    for x, y, z in itertools.product(range(Nx), range(Ny), range(Nz)):
        grid.add_edge((x, y, z), ((x + 1) % Nx, y, z), axis="x")
        grid.add_edge((x, y, z), (x, (y + 1) % Ny, z), axis="y")
        grid.add_edge((x, y, z), (x, y, (z + 1) % Nz), axis="z")

    # Relabel 2D position to corresponding indices
    pos2idx = {
        (x, y, z): i
        for i, (z, y, x) in enumerate(
            itertools.product(range(Nz), range(Ny), range(Nx))
        )
    }
    grid = nx.relabel.relabel_nodes(grid, pos2idx)

    # Sort node idx
    graph = nx.DiGraph()
    graph.add_nodes_from(range(grid.number_of_nodes()))
    graph.add_edges_from(grid.edges)

    # Remove self-loop
    graph.remove_edges_from(nx.function.selfloop_edges(graph))
    return graph


def get_square_grid(*dims: int) -> nx.DiGraph:
    """Get graph representation of square mesh with periodic boundary condition"""
    if len(dims) == 1:
        return get_1d_square_grid(*dims)
    elif len(dims) == 2:
        return get_2d_square_grid(*dims)
    elif len(dims) == 3:
        return get_3d_square_grid(*dims)
    raise ValueError
