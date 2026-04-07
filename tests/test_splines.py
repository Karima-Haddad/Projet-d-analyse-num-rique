import pytest
from src.splines.linear_spline import linear_spline
from src.splines.cubic_spline import natural_cubic_spline


def test_linear_spline_midpoint() -> None:
    x = [0.0, 1.0]
    y = [0.0, 2.0]
    assert linear_spline(x, y, 0.5) == 1.0


def test_cubic_spline_exact_points() -> None:
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 0.0]
    assert natural_cubic_spline(x, y, 1.0) == pytest.approx(1.0)
