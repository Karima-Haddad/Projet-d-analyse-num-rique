"""
Visualization module for numerical analysis.
"""

import matplotlib.pyplot as plt
from typing import Sequence, Dict, Callable


class Visualizer:
    """
    Class for plotting and visualizing numerical analysis results.
    """

    def __init__(self, style: str = "seaborn-v0_8-darkgrid", figsize=(10, 6)):
        plt.style.use(style)
        self.figsize = figsize

    def plot_data(self, x: Sequence[float], y: Sequence[float], title: str = "Plot") -> None:
        plt.figure(figsize=self.figsize)
        plt.plot(x, y, marker="o")
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_interpolation_comparison(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        interpolators: Dict[str, Callable],
        x_fine: Sequence[float],
        title: str = "Interpolation Comparison"
    ) -> None:
        plt.figure(figsize=self.figsize)

        plt.scatter(x_data, y_data, color="red", label="Data")

        for name, func in interpolators.items():
            y_vals = [func(x) for x in x_fine]
            plt.plot(x_fine, y_vals, label=name)

        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_runge_phenomenon(
        self,
        x_fine: Sequence[float],
        y_true: Sequence[float],
        interpolations: Dict[str, Sequence[float]]
    ) -> None:
        plt.figure(figsize=self.figsize)

        plt.plot(x_fine, y_true, label="True function", linewidth=2)

        for name, y_vals in interpolations.items():
            plt.plot(x_fine, y_vals, linestyle="--", label=name)

        plt.title("Runge Phenomenon")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_convergence(
        self,
        n_values: Sequence[int],
        errors: Dict[str, Sequence[float]],
        methods: Sequence[str],
        title: str = "Convergence"
    ) -> None:
        plt.figure(figsize=self.figsize)

        for method in methods:
            plt.plot(n_values, errors[method], marker="o", label=method)

        plt.yscale("log")
        plt.xlabel("n")
        plt.ylabel("Error (log scale)")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_cooling_analysis(
        self,
        t_data: Sequence[float],
        T_data: Sequence[float],
        t_fine: Sequence[float],
        T_interp: Sequence[float],
        k_opt: float,
        T_model: Sequence[float]
    ) -> None:
        plt.figure(figsize=self.figsize)

        plt.scatter(t_data, T_data, color="red", label="Data")
        plt.plot(t_fine, T_interp, label="Interpolation")
        plt.plot(t_fine, T_model, label=f"Model (k={k_opt:.4f})")

        plt.title("Cooling Analysis")
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_flow_analysis(
        self,
        x_data: Sequence[float],
        v_data: Sequence[float],
        x_fine: Sequence[float],
        v_interp: Sequence[float],
        w_function: Callable
    ) -> None:
        plt.figure(figsize=self.figsize)

        plt.scatter(x_data, v_data, color="red", label="Velocity data")
        plt.plot(x_fine, v_interp, label="Interpolated velocity")

        w_vals = [w_function(x) for x in x_fine]
        plt.plot(x_fine, w_vals, linestyle="--", label="Width function")

        plt.title("Flow Analysis")
        plt.xlabel("x")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
