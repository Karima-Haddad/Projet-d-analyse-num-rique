from typing import Callable
import numpy as np


def gauss_legendre(f: Callable[[float], float], a: float, b: float, n: int = 3) -> float:
    x, w = np.polynomial.legendre.leggauss(n)
    midpoint = 0.5 * (a + b)
    half_range = 0.5 * (b - a)
    return half_range * sum(w[i] * f(midpoint + half_range * x[i]) for i in range(n))
