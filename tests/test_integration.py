import pytest
from src.integration.newton_cotes import simpson
from src.integration.adaptive_integration import adaptive_trapezoid
from src.integration.gauss_quadrature import gauss_legendre


def test_simpson_quadratic() -> None:
    f = lambda x: x**2
    assert simpson(f, 0.0, 1.0, n=10) == pytest.approx(1.0 / 3.0, rel=1e-4)


def test_adaptive_trapezoid_constant() -> None:
    f = lambda x: 5.0
    assert adaptive_trapezoid(f, 0.0, 1.0) == pytest.approx(5.0)


def test_gauss_legendre_linear() -> None:
    f = lambda x: 2.0 * x
    assert gauss_legendre(f, 0.0, 1.0, n=3) == pytest.approx(1.0)
