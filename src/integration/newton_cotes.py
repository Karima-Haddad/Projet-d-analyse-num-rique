from typing import Callable

def simpson(f: Callable[[float], float], a: float, b: float, n: int = 100) -> float:
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        weight = 4 if i % 2 else 2
        total += weight * f(a + i * h)
    return total * h / 3.0
