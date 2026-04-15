import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.integration.newton_cotes import NewtonCotes


# --------------------------------------------------
# Test précision sur fonction connue
# --------------------------------------------------

def test_simpson_exp():
    result = NewtonCotes.simpson(np.exp, 0, 1, n=100)
    exact = np.e - 1
    assert abs(result - exact) < 1e-6


def test_trapezoidal_exp():
    result = NewtonCotes.trapezoidal(np.exp, 0, 1, n=1000)
    exact = np.e - 1
    assert abs(result - exact) < 1e-4


def test_rectangle_exp():
    result = NewtonCotes.rectangle(np.exp, 0, 1, n=1000)
    exact = np.e - 1
    assert abs(result - exact) < 1e-4


def test_simpson_38_exp():
    result = NewtonCotes.simpson_38(np.exp, 0, 1, n=99)
    exact = np.e - 1
    assert abs(result - exact) < 1e-6


# --------------------------------------------------
# Test avec fonctions simples
# --------------------------------------------------

def test_polynomial():
    f = lambda x: x**2
    result = NewtonCotes.simpson(f, 0, 1, n=100)
    assert abs(result - 1/3) < 1e-6


def test_sin():
    result = NewtonCotes.simpson(np.sin, 0, np.pi, n=100)
    assert abs(result - 2) < 1e-6


# --------------------------------------------------
# Test erreurs (IMPORTANT)
# --------------------------------------------------

def test_simpson_odd_n():
    with pytest.raises(ValueError):
        NewtonCotes.simpson(np.exp, 0, 1, n=3)


def test_simpson_38_invalid_n():
    with pytest.raises(ValueError):
        NewtonCotes.simpson_38(np.exp, 0, 1, n=4)


def test_invalid_interval():
    with pytest.raises(ValueError):
        NewtonCotes.trapezoidal(np.exp, 1, 0, n=10)


def test_invalid_n():
    with pytest.raises(ValueError):
        NewtonCotes.rectangle(np.exp, 0, 1, n=0)


def test_errors_display():
    import numpy as np
    from src.integration.newton_cotes import NewtonCotes

    f = np.exp
    a, b = 0, 1
    exact = np.e - 1

    methods = {
        "Rectangle": NewtonCotes.rectangle,
        "Trapèze": NewtonCotes.trapezoidal,
        "Simpson": NewtonCotes.simpson,
        "Simpson 3/8": NewtonCotes.simpson_38
    }

    print("\n--- Comparaison des erreurs ---")

    for name, method in methods.items():
        if name == "Simpson 3/8":
            approx = method(f, a, b, n=99)
        else:
            approx = method(f, a, b, n=100)

        error = abs(approx - exact)

        print(f"{name:<15} | approx = {approx:.10f} | erreur = {error:.2e}")

    # Juste pour que pytest passe
    assert True