import numpy as np
import pytest
from src.applications.FlowProblem import FlowProblem


# Données du sujet
x_data = np.array([0, 0.5, 1.2, 1.8, 2.5, 3.1, 3.7, 4.2, 4.8, 5.3, 6.0])
v_data = np.array([0, 2.1, 3.8, 5.2, 6.4, 7.0, 7.3, 7.2, 6.8, 5.9, 4.5])


def test_velocity():
    fp = FlowProblem(x_data, v_data)

    # doit retrouver les points connus
    assert abs(fp.velocity(0) - 0) < 1e-6
    assert abs(fp.velocity(6.0) - 4.5) < 1e-6


def test_local_flow_rate():
    fp = FlowProblem(x_data, v_data)

    q = fp.local_flow_rate(0)
    expected = 0 * (0.5 + 0.1 * 0)

    assert abs(q - expected) < 1e-6


def test_total_flow_rate():
    fp = FlowProblem(x_data, v_data)

    D1 = fp.total_flow_rate(method="adaptive")
    D2 = fp.total_flow_rate(method="simpson", n=100)

    # Les deux méthodes doivent donner des résultats proches
    assert abs(D1 - D2) < 1e-2


def test_acceleration():
    fp = FlowProblem(x_data, v_data)

    a = fp.acceleration(3.0)

    # juste vérifier que ça retourne une valeur valide
    assert np.isfinite(a)


def test_work():
    fp = FlowProblem(x_data, v_data)

    W = fp.work(mass=2.0)

    # Travail doit être positif
    assert W > 0


def test_invalid_mass():
    fp = FlowProblem(x_data, v_data)

    with pytest.raises(ValueError):
        fp.work(mass=0)