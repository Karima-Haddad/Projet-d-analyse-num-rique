from src.applications.cooling_problem import cooling_temperature
from src.applications.flow_problem import flow_rate


def main() -> None:
    t = 5.0
    print(f"Cooling temperature after {t} seconds: {cooling_temperature(t):.2f}°C")
    print(f"Pipe flow rate at 10 m/s: {flow_rate(10.0):.2f} m^3/s")


if __name__ == "__main__":
    main()
