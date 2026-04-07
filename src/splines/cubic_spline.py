from typing import Sequence
import numpy as np


def natural_cubic_spline(x: Sequence[float], y: Sequence[float], x_eval: float) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same length")
    if len(x_arr) < 3:
        raise ValueError("At least three points are required for cubic spline")

    n = len(x_arr)
    h = np.diff(x_arr)
    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = (3.0 / h[i]) * (y_arr[i + 1] - y_arr[i]) - (3.0 / h[i - 1]) * (y_arr[i] - y_arr[i - 1])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        l[i] = 2.0 * (x_arr[i + 1] - x_arr[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    b = np.zeros(n - 1)
    c = np.zeros(n)
    d = np.zeros(n - 1)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y_arr[j + 1] - y_arr[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    if x_eval <= x_arr[0]:
        i = 0
    elif x_eval >= x_arr[-1]:
        i = n - 2
    else:
        i = np.searchsorted(x_arr, x_eval) - 1

    dx = x_eval - x_arr[i]
    return float(y_arr[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3)
