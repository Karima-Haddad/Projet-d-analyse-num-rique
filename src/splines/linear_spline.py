from typing import Sequence
import numpy as np


def linear_spline(x: Sequence[float], y: Sequence[float], x_eval: float) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same length")

    if x_eval <= x_arr[0]:
        return float(y_arr[0])
    if x_eval >= x_arr[-1]:
        return float(y_arr[-1])

    for i in range(len(x_arr) - 1):
        if x_arr[i] <= x_eval <= x_arr[i + 1]:
            t = (x_eval - x_arr[i]) / (x_arr[i + 1] - x_arr[i])
            return float(y_arr[i] * (1 - t) + y_arr[i + 1] * t)

    raise ValueError("x_eval is out of bounds")
