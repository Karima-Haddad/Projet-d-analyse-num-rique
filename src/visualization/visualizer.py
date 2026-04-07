import matplotlib.pyplot as plt
from typing import Sequence


def plot_data(x: Sequence[float], y: Sequence[float], title: str = "Plot") -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
