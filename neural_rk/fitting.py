import numpy as np
import numpy.typing as npt

arr = npt.NDArray[np.float32]


def log_fit(
    raw_x: arr,
    raw_y: arr,
    start: float | None = None,
    end: float | None = None,
    offset: float = 0.0,
) -> tuple[arr, arr, float, float]:
    """
    log-log scale linear fitting of (raw_x, raw_y).
    return fit_x, fit_y, slope, residual
    """
    if not isinstance(raw_x, np.ndarray):
        raw_x = np.array(raw_x, dtype=np.float64)
    if not isinstance(raw_y, np.ndarray):
        raw_y = np.array(raw_y, dtype=np.float64)
    assert len(raw_x) == len(raw_y), "log_fit: size mismatch"
    assert raw_x.dtype == raw_y.dtype, "log_fit: dtype mismatch"

    start = raw_x.min() if start is None else start
    end = raw_x.max() if end is None else end

    poly, residual, *_ = np.polynomial.Polynomial.fit(
        np.log10(raw_x), np.log10(raw_y), deg=1, full=True
    )
    coeff: list[float] = poly.convert().coef[::-1]
    fit_x = np.array([start, end], dtype=raw_x.dtype)
    fit_y = 10.0 ** (coeff[1] - offset) * np.power(fit_x, coeff[0], dtype=raw_x.dtype)
    return fit_x, fit_y, coeff[0], residual[0][0]


def lin_log_fit(
    raw_x: arr,
    raw_y: arr,
    start: float | None = None,
    end: float | None = None,
    offset: float = 0.0,
) -> tuple[arr, arr, float, float]:
    """raw_x scale is linear, raw_y scale is log"""
    if not isinstance(raw_x, np.ndarray):
        raw_x = np.array(raw_x, dtype=np.float64)
    if not isinstance(raw_y, np.ndarray):
        raw_y = np.array(raw_y, dtype=np.float64)
    assert len(raw_x) == len(raw_y), "lin_log_fit: size mismatch"
    assert raw_x.dtype == raw_y.dtype, "lin_log_fit: dtype mismatch"

    start = raw_x.min() if start is None else start
    end = raw_x.max() if end is None else end

    poly, residual, *_ = np.polynomial.Polynomial.fit(
        raw_x, np.log10(raw_y), deg=1, full=True
    )
    coeff: list[float] = poly.convert().coef[::-1]
    fit_x = np.array([start, end], dtype=raw_x.dtype)
    fit_y = np.power(10.0, coeff[0] * fit_x + coeff[1] - offset)
    return fit_x, fit_y, coeff[0], residual[0][0]


def log_lin_fit(
    raw_x: arr,
    raw_y: arr,
    start: float | None = None,
    end: float | None = None,
    offset: float = 0.0,
) -> tuple[arr, arr, float, float]:
    """raw_x scale is log, raw_y scale is linear"""
    if not isinstance(raw_x, np.ndarray):
        raw_x = np.array(raw_x, dtype=np.float64)
    if not isinstance(raw_y, np.ndarray):
        raw_y = np.array(raw_y, dtype=np.float64)
    assert len(raw_x) == len(raw_y), "log_lin_fit: size mismatch"
    assert raw_x.dtype == raw_y.dtype, "log_lin_fit: dtype mismatch"

    start = raw_x.min() if start is None else start
    end = raw_x.max() if end is None else end

    poly, residual, *_ = np.polynomial.Polynomial.fit(
        np.log10(raw_x), raw_y, deg=1, full=True
    )
    coeff: list[float] = poly.convert().coef[::-1]
    fit_x = np.array([start, end], dtype=raw_x.dtype)
    fit_y = coeff[0] * np.log10(fit_x) + coeff[1] - offset
    return fit_x, fit_y, coeff[0], residual[0][0]


def lin_fit(
    raw_x: arr,
    raw_y: arr,
    start: float | None = None,
    end: float | None = None,
    offset: float = 0.0,
) -> tuple[arr, arr, float, float]:
    """raw_x scale and raw_y scale are linear"""
    if not isinstance(raw_x, np.ndarray):
        raw_x = np.array(raw_x, dtype=np.float64)
    if not isinstance(raw_y, np.ndarray):
        raw_y = np.array(raw_y, dtype=np.float64)
    assert len(raw_x) == len(raw_y), "lin_fit: size mismatch"
    assert raw_x.dtype == raw_y.dtype, "lin_fit: dtype mismatch"

    start = raw_x.min() if start is None else start
    end = raw_x.max() if end is None else end

    poly, residual, *_ = np.polynomial.Polynomial.fit(raw_x, raw_y, deg=1, full=True)
    coeff: list[float] = poly.convert().coef[::-1]
    fit_x = np.array([start, end], dtype=raw_x.dtype)
    fit_y = coeff[0] * fit_x + coeff[1] - offset
    return fit_x, fit_y, coeff[0], residual
