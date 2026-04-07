from typing import Sequence
import numpy as np


def lagrange_polynomial(x: Sequence[float], y: Sequence[float], x_eval: float) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same length")

    total = 0.0
    n = len(x_arr)
    for i in range(n):
        term = y_arr[i]
        for j in range(n):
            if i != j:
                term *= (x_eval - x_arr[j]) / (x_arr[i] - x_arr[j])
        total += term
    return float(total)
