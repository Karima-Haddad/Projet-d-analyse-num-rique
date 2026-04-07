from typing import Callable


def adaptive_trapezoid(f: Callable[[float], float], a: float, b: float, tol: float = 1e-6) -> float:
    mid = (a + b) / 2.0
    left = (f(a) + f(mid)) * (mid - a) / 2.0
    right = (f(mid) + f(b)) * (b - mid) / 2.0
    whole = (f(a) + f(b)) * (b - a) / 2.0
    if abs(left + right - whole) < tol:
        return left + right
    return adaptive_trapezoid(f, a, mid, tol / 2) + adaptive_trapezoid(f, mid, b, tol / 2)
