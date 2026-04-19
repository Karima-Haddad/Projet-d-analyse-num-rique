"""
Visualization module for numerical analysis.
"""

from typing import Callable, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """
    Class for plotting and visualizing numerical analysis results.
    """

    def __init__(
        self,
        style: str = "seaborn-v0_8-darkgrid",
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        """
        Initialize the visualizer.

        Parameters
        ----------
        style : str, optional
            Matplotlib style to use.
        figsize : tuple[int, int], optional
            Default figure size.
        """
        plt.style.use(style)
        self.figsize = figsize

    def _finalize(self, save_path: str | None = None) -> None:
        """
        Finalize the current figure.

        Parameters
        ----------
        save_path : str or None, optional
            Path where the figure should be saved.
        """
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def plot_data(
        self,
        x: Sequence[float],
        y: Sequence[float],
        title: str = "Plot",
        save_path: str | None = None,
    ) -> None:
        """
        Plot basic x-y data.
        """
        plt.figure(figsize=self.figsize)
        plt.plot(x, y, marker="o")
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        self._finalize(save_path)

    def plot_interpolation_comparison(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        interpolators: Dict[str, Callable],
        x_fine: Sequence[float],
        title: str = "Interpolation Comparison",
        save_path: str | None = None,
    ) -> None:
        """
        Plot interpolation curves and original data points.
        """
        plt.figure(figsize=self.figsize)

        plt.scatter(x_data, y_data, color="red", label="Data", zorder=3)

        for name, func in interpolators.items():
            y_vals = np.asarray([func(x) for x in x_fine], dtype=float)
            plt.plot(x_fine, y_vals, linewidth=2, label=name)

        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        self._finalize(save_path)

    def plot_runge_phenomenon(
        self,
        x_fine: Sequence[float],
        y_true: Sequence[float],
        interpolations: Dict[str, Sequence[float]],
        save_path: str | None = None,
    ) -> None:
        """
        Plot the Runge function and its interpolations.
        """
        plt.figure(figsize=self.figsize)

        plt.plot(x_fine, y_true, label="True function", linewidth=2.5)

        for name, y_vals in interpolations.items():
            plt.plot(x_fine, y_vals, linestyle="--", linewidth=2, label=name)

        plt.title("Runge Phenomenon")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        self._finalize(save_path)

    def plot_convergence(
        self,
        n_values: Sequence[int],
        errors: Dict[str, Sequence[float]],
        methods: Sequence[str],
        title: str = "Convergence",
        save_path: str | None = None,
    ) -> None:
        """
        Plot convergence curves for numerical integration methods.
        """
        plt.figure(figsize=self.figsize)

        for method in methods:
            plt.plot(n_values, errors[method], marker="o", linewidth=2, label=method)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("n")
        plt.ylabel("Error")
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both")
        self._finalize(save_path)

    def plot_cooling_analysis(
        self,
        t_data: Sequence[float],
        T_data: Sequence[float],
        t_fine: Sequence[float],
        T_interp: Sequence[float],
        k_opt: float,
        T_model: Sequence[float],
        save_path: str | None = None,
    ) -> None:
        """
        Plot cooling data, interpolation, and exponential model.
        """
        plt.figure(figsize=self.figsize)

        plt.scatter(t_data, T_data, color="red", label="Data", zorder=3)
        plt.plot(t_fine, T_interp, linewidth=2, label="Interpolation")
        plt.plot(
            t_fine,
            T_model,
            linewidth=2,
            linestyle="--",
            label=f"Model (k={k_opt:.4f})",
        )

        plt.title("Cooling Analysis")
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.legend()
        plt.grid(True)
        self._finalize(save_path)

    def plot_flow_analysis(
        self,
        x_data: Sequence[float],
        v_data: Sequence[float],
        x_fine: Sequence[float],
        v_interp: Sequence[float],
        w_function: Callable,
        save_path: str | None = None,
    ) -> None:
        """
        Plot flow velocity data, interpolated velocity, and width function.
        """
        plt.figure(figsize=self.figsize)

        plt.scatter(x_data, v_data, color="red", label="Velocity data", zorder=3)
        plt.plot(x_fine, v_interp, linewidth=2, label="Interpolated velocity")

        w_vals = np.asarray([w_function(x) for x in x_fine], dtype=float)
        plt.plot(x_fine, w_vals, linestyle="--", linewidth=2, label="Width function")

        plt.title("Flow Analysis")
        plt.xlabel("x")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        self._finalize(save_path)