from src.applications.cooling_problem import cooling_temperature
from src.applications.flow_problem import flow_rate


def test_cooling_temperature_at_zero() -> None:
    assert cooling_temperature(0.0) == pytest.approx(100.0)


def test_flow_rate_default_area() -> None:
    assert flow_rate(10.0) == 1.0
