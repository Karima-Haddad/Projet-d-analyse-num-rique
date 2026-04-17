# Algo Numérique

Projet Python d'analyse numérique dédié à l'interpolation polynomiale, à l'intégration numérique et à l'étude de deux problèmes appliqués : refroidissement et écoulement.

## Description

Ce projet propose un ensemble de modules pour :
- construire et évaluer des interpolations polynomiales par Lagrange et Newton,
- implémenter des règles de quadrature de Newton-Cotes (rectangle, trapèze, Simpson),
- réaliser une intégration adaptative de type Simpson,
- modéliser un problème de refroidissement à partir de données expérimentales,
- analyser un problème d'écoulement dans un canal à largeur variable.

## Fonctionnalités principales

- `src.interpolation.polynomial_interpolation.PolynomialInterpolation`
  - interpolation par la formule de Lagrange,
  - interpolation par la forme de Newton avec différences divisées.
- `src.integration.newton_cotes.NewtonCotes`
  - intégration numérique composée : rectangle, trapèze, Simpson.
- `src.integration.adaptive.AdaptiveIntegration`
  - intégration adaptative basée sur la méthode de Simpson.
- `src.applications.cooling_problem.CoolingProblem`
  - analyse de données de refroidissement,
  - estimation d'un modèle exponentiel,
  - calcul de pertes thermiques.
- `src.applications.FlowProblem.FlowProblem`
  - interpolation de profils de vitesse,
  - calcul du débit volumique,
  - estimation d'accélération spatiale.
- `src.visualization.visualizer.Visualizer`
  - visualisation des données, des interpolations et des phénomènes numériques.

## Structure du projet

- `main.py` : point d'entrée pour exécuter des démonstrations et visualiser les résultats.
- `src/` : implémentation des modules d'interpolation, d'intégration, d'applications et de visualisation.
- `tests/` : tests unitaires pour valider les fonctions et classes.
- `data/` : données d'exemple et fichiers de référence.
- `results/` : répertoires de sortie pour les figures et les résultats.
- `docs/` : notes et documentation de projet.

## Prérequis

- Python 3.10 ou supérieur
- `numpy`
- `scipy`
- `matplotlib`
- `pytest`

## Installation

1. Créez un environnement virtuel :

```bash
python -m venv .venv
```

2. Activez l'environnement :

- Windows :
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- macOS/Linux :
  ```bash
  source .venv/bin/activate
  ```

3. Installez les dépendances :

```bash
python -m pip install -r requirements.txt
```

## Exécution

Lancez le script principal pour exécuter les démonstrations :

```bash
python -m main
```


## Utilisation rapide

Exemple d'interpolation polynomiale :

```python
from src.interpolation.polynomial_interpolation import PolynomialInterpolation

x_points = [0, 1, 2]
y_points = [1, 3, 2]
interp = PolynomialInterpolation(x_points, y_points)
print(interp.evaluate(1.5, method="newton"))
```

Exemple d'analyse d'un problème de refroidissement :

```python
from src.applications.cooling_problem import CoolingProblem
import numpy as np

cooling = CoolingProblem(
    t_data=np.array([0, 1, 2, 3]),
    T_data=np.array([90, 85, 72, 63]),
    T_ambient=20.0,
    h_coeff=50.0,
)
print(cooling.total_heat_loss(method="adaptive"))
```

## Notes

- Le fichier `requirements.txt` contient les bibliothèques Python nécessaires.
- Le répertoire `results/` est prévu pour stocker les sorties générées par le projet.
- Pour étendre le projet, ajoutez de nouveaux cas d'études dans `src/applications` et des tests correspondants dans `tests/`.

