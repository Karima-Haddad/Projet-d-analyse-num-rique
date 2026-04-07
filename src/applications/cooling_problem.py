import numpy as np


def cooling_temperature(t: float, T_env: float = 20.0, T0: float = 100.0, k: float = 0.1) -> float:
    return float(T_env + (T0 - T_env) * np.exp(-k * t))
