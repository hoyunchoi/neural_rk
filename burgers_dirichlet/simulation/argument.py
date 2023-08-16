import argparse
from typing import Callable

import numpy as np
import numpy.typing as npt

arr = npt.NDArray[np.float32]


def get_args(options: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the dataset")

    # * Grid parameters
    parser.add_argument(
        "--Nx",
        type=int,
        nargs="+",
        default=[100],
        help=(
            "Number of grid points at x-axis. If given single value, it is constant. If"
            " given 2 values, uniform random between two values (inclusive). If given 3"
            " or more values, choose among them."
        ),
    )
    parser.add_argument(
        "--Ny",
        type=int,
        nargs="*",
        default=[100],
        help=(
            "Number of grid points at y-axis. If the value is not given or the value is"
            " 1, run 1D burgers simulation. If given sigle value, it is constant. If"
            " given 2 values, uniform random between two values (inclusive). If given 3"
            " or more values, choose among them."
        ),
    )
    parser.add_argument(
        "--dx_delta",
        type=float,
        default=0.0,
        help=(
            "Delta percentile for grid spacings of both x, y axis. See"
            " argument.divide_randomly"
        ),
    )
    parser.add_argument(
        "--dx_clip",
        nargs=4,
        type=float,
        default=[-1.0, -1.0, -1.0, -1.0],
        help="Clip for grid spacings of both x, y axis. See argument.divde_randomly",
    )

    # * Initial condition
    parser.add_argument(
        "--num_cycles_ux",
        type=int,
        nargs="+",
        default=[1],
        help=(
            "Number of cycle(periods) for field u, direction x. If the given single"
            " value, constant over all samples.  If given 2 values, uniform random"
            " between two values (inclusive). If given 3 or more values, choose among"
            " them."
        ),
    )
    parser.add_argument(
        "--num_cycles_uy",
        type=int,
        nargs="+",
        default=[1],
        help=(
            "Number of cycle(periods) for field u, direction y. If the given single"
            " value, constant over all samples.  If given 2 values, uniform random"
            " between two values. (inclusive) If given 3 or more values, choose among"
            " them."
        ),
    )
    parser.add_argument(
        "--num_cycles_vx",
        type=int,
        nargs="+",
        default=[1],
        help=(
            "Number of cycle(periods) for field v, direction x. If the given single"
            " value, constant over all samples.  If given 2 values, uniform random"
            " between two values (inclusive). If given 3 or more values, choose among"
            " them."
        ),
    )
    parser.add_argument(
        "--num_cycles_vy",
        type=int,
        nargs="+",
        default=[1],
        help=(
            "Number of cycle(periods) for field v, direction y. If the given single"
            " value, constant over all samples.  If given 2 values, uniform random"
            " between two values (inclusive). If given 3 or more values, choose among"
            " them."
        ),
    )

    # * Parameters for burgers equation
    parser.add_argument(
        "--nu",
        type=float,
        nargs="+",
        default=[0.01],
        help=(
            "dissipation rate. If single value is given, it is constant over every"
            " edges. If two values are given, uniform random between two values"
            " (inclusive). If 3 or more values are given, choose among them."
        ),
    )

    # * Simulation parameters
    parser.add_argument(
        "--solver",
        choices=["rk1", "rk2", "rk4", "rk1_graph", "rk2_graph", "rk4_graph"],
        default="rk4",
        help="Which Runge-Kutta method to solve equation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help=(
            "Number of time steps per simulation. dt is determined by the average of"
            " max_time / steps."
        ),
    )
    parser.add_argument(
        "--dt_delta",
        type=float,
        default=0.0,
        help="Delta percentile for dt. See argument.divide_randomly",
    )
    parser.add_argument(
        "--dt_clip",
        type=float,
        nargs=2,
        default=[-1.0, -1.0],
        help="Clip for dt. See argument.divide_randomly",
    )

    # * constant over samples
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples",
    )
    parser.add_argument(
        "--const_graph",
        action="store_true",
        help="If this flag is on, graph is constant over samples",
    )
    parser.add_argument(
        "--const_dt",
        action="store_true",
        help="If this flag is on, dt is constant over samples",
    )
    parser.add_argument(
        "--const_param",
        action="store_true",
        help="If this flag is on, params(nu) is constant over samples",
    )
    parser.add_argument(
        "--const_ic",
        action="store_true",
        help="If this flag is on, initial condition is constant over samples",
    )

    # * Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If given, seed the random engine for reproducibility",
    )
    parser.add_argument(
        "--seed_ic",
        type=int,
        default=None,
        help=(
            "If given, create new random engine only for initial condition. If not"
            " given use default random engine"
        ),
    )

    # Parse the arguments and return
    if options is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args=options)


def divide_randomly(
    total: float,
    size: int,
    delta_percentile: float,
    clip: tuple[float | None, float | None],
    rng: np.random.Generator,
    eps: float = 1e-5,
) -> arr:
    """
    Find array of floats with sum of total whose distribution is uniform

    Args
    total: sum(return) = total
    size: number of random numbers
    delta_percentile: return value lies between [total / size - delta, total / size + delta]
                      where delta = avg * delta_percentile
    clip: Range of returning values. Will override delta range if the range is narrower then delta

    Return
    floats: [size, ] where sum(floats) = total
    """
    assert 0.0 <= delta_percentile <= 1.0

    # Distribution setting
    avg = total / size
    delta = avg * delta_percentile
    low_range, high_range = avg - delta, avg + delta
    if clip[0] is not None:
        low_range = max(clip[0], low_range)
    if clip[1] is not None:
        high_range = min(clip[1], high_range)
    if total + eps < size * low_range or total - eps > size * high_range:
        raise ValueError("No sulution exists with given parameters.")

    # Find random floats with sum of total
    numbers = np.zeros(size, dtype=np.float32)
    remaining = total
    for i in range(size):
        # Maximum/minimum range of current number
        high = min(high_range, remaining - (size - i - 1) * low_range)
        low = max(low_range, remaining - (size - i - 1) * high_range)

        # Randomly select number
        value = rng.uniform(min(low, high), max(low, high))
        numbers[i] = value
        remaining -= value

    rng.shuffle(numbers)
    return numbers


def get_NxNy(
    args_Nx: list[int], args_Ny: list[int], rng: np.random.Generator
) -> tuple[int, int]:
    """
    Randomly choose number of grid points Nx, Ny of each axis x,y

    Args
    args_Nx: Nx will be randomly selected between the values
    args_Ny: Ny will be randomly selected between the values

    Return
    Nx, Ny: Number of grid point of each axis x, y
    """
    if len(args_Nx) <= 2:
        Nx = rng.integers(min(args_Nx), max(args_Nx), endpoint=True)
    else:
        Nx = rng.choice(args_Nx)

    if len(args_Ny) <= 2:
        Ny = rng.integers(min(args_Ny), max(args_Ny), endpoint=True)
    else:
        Ny = rng.choice(args_Ny)
    return Nx, Ny


def get_dxdy(
    LxLy: tuple[float, float],
    NxNy: tuple[int, int],
    delta_percentile: float,
    clip: tuple[float | None, float | None, float | None, float | None],
    rng: np.random.Generator,
) -> tuple[arr, arr]:
    """
    Randomly create grid spacing of given number of Nx,Ny inside domain of length LxLy

    Args
    LxLy: Length of each axis x, y
    NxNy: Number of grid points of each axis x,y
    delta_percentile: see divide_randomly
    clip: see divide_randomly

    Return
    dx, dy: [Nx-1, ], [Ny-1, ], Spacing of each axis
    """
    Lx, Ly = LxLy
    Nx, Ny = NxNy

    # Make negative value to None
    clip = tuple(map(lambda x: x if isinstance(x, float) and x > 0 else None, clip))

    # Sample dx, dy
    dx = divide_randomly(Lx, Nx - 1, delta_percentile, clip[:2], rng=rng)
    dy = divide_randomly(Ly, Ny - 1, delta_percentile, clip[2:], rng=rng)

    return dx, dy


def get_dt(
    max_time: float,
    steps: int,
    delta_percentile: float,
    clip: tuple[float | None, float | None],
    rng: np.random.Generator,
) -> arr:
    """
    Randomly create time spacing of given number of steps for given max_time

    Args
    max_time: maximum time will be randomly selected between the values
    steps: Number of time points
    delta_percentile: see divide_randomly
    clip: see divide_randomly

    Return
    dt: [S, 1]
    """
    # Make negative value to None
    clip = tuple(map(lambda x: x if isinstance(x, float) and x > 0 else None, clip))

    # Sample dt
    return divide_randomly(max_time, steps, delta_percentile, clip, rng)[:, None]


def get_nu(
    ndim: int, args_nu: list[float], rng: np.random.Generator
) -> tuple[float, float]:
    """
    Randomly select nu in the range of args_nu, distribution: uniform / log-uniform

    Args
    ndim: dimension, 1 or 2
    args_nu: nu will be randomly selected between values

    Return
    nu_x, nu_y
    e.g., ndim=1 -> (nu, 0.0), ndim=2 -> [nu, nu]
    """
    if len(args_nu) <= 2:
        nu = rng.uniform(min(args_nu), max(args_nu))
        # nu = np.power(10.0, rng.uniform(np.log10(min(args_nu)), np.log10(max(args_nu))))
    else:
        nu = rng.choice(args_nu)

    if ndim == 1:
        return nu, 0.0
    else:
        return nu, nu


def get_initial_condition(
    LxLy: tuple[float, float],
    ndim: int,
    num_cycles_ux: list[int],
    num_cycles_uy: list[int],
    num_cycles_vx: list[int],
    num_cycles_vy: list[int],
    rng: np.random.Generator,
) -> Callable[[arr], arr]:
    """
    Create 2D periodic sin assignment function

    Args
    LxLy: Length of each axis x, y
    ndim: dimension, 1 or 2
    num_cycles: Number of cycles (periods) of field (u, v) at each axis (x,y)

    Return
    function with
        Args: position [Ny, Nx, 2]
        Return: initial condition [Ny, Nx, 2], u,v of each grid point
    """

    # Period of each field (u, v) and axis (x, y)
    Lx, Ly = LxLy
    period_ux = Lx / rng.integers(min(num_cycles_ux), max(num_cycles_ux), endpoint=True)
    period_uy = Ly / rng.integers(min(num_cycles_uy), max(num_cycles_uy), endpoint=True)
    period_vx = Lx / rng.integers(min(num_cycles_vx), max(num_cycles_vx), endpoint=True)
    period_vy = Ly / rng.integers(min(num_cycles_vy), max(num_cycles_vy), endpoint=True)

    # Offset of each field (u, v) and axis (x, y)
    offset = rng.uniform(0.0, 1.0, size=(4,)).astype(np.float32)

    def asaymmetric_sin_2d(position: arr) -> arr:
        def sin(x: arr, period: float, phase: float = 0.0) -> arr:
            return np.sin(2.0 * np.pi / period * x - phase)

        x, y = position[..., 0], position[..., 1]
        if ndim == 1:
            asymmetry = np.exp(-((x - offset[0]) ** 2) / rng.uniform(0.1, 0.5))
            initial_u = sin(x, period_ux) * asymmetry
            initial_v = np.zeros_like(initial_u)  # zero-field for 1d
        else:
            asymmetry_u = np.exp(
                -((x - offset[0]) ** 2 + (y - offset[1]) ** 2) / rng.uniform(0.1, 0.5)
            )
            asymmetry_v = np.exp(
                -((x - offset[2]) ** 2 + (y - offset[3]) ** 2) / rng.uniform(0.1, 0.5)
            )
            initial_u = sin(x, period_ux) * sin(y, period_uy) * asymmetry_u
            initial_v = sin(x, period_vx) * sin(y, period_vy) * asymmetry_v

        initial_u /= np.abs(initial_u).max()
        initial_v /= np.abs(initial_v).max()

        return np.stack((initial_u, initial_v), axis=-1)

    return asaymmetric_sin_2d
