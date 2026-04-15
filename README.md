# Algo Numerique

This project is a numerical analysis toolkit for interpolation, splines, integration, and applied problems.

## Structure

- `main.py` - entry point for simple demonstrations.
- `src/` - implementation modules.
- `tests/` - unit tests for the numerical routines.
- `data/` - example data files.
- `results/` - placeholder results output directories.
- `docs/` - documentation and notes.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Running tests

```bash
pytest
```
## Interpolation polynomiale

La classe `PolynomialInterpolation` permet :
- de construire un polynôme interpolateur à partir de points donnés,
- de l’évaluer avec la méthode de Lagrange,
- de calculer les coefficients de Newton par différences divisées,
- de l’évaluer avec la forme de Newton.

### Exemple
```python
from src.interpolation.polynomial_interpolation import PolynomialInterpolation

x_points = [0, 1, 2]
y_points = [1, 3, 2]

interp = PolynomialInterpolation(x_points, y_points)
print(interp.evaluate(1.5, method="newton"))
