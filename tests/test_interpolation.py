import pytest
from src.interpolation.polynomial_interpolation import lagrange_polynomial


def test_lagrange_polynomial_linear() -> None:
    x = [0.0, 1.0]
    y = [1.0, 3.0]
    assert lagrange_polynomial(x, y, 0.5) == pytest.approx(2.0)
