"""
Module pour l'interpolation polynomiale.

Ce module contient la classe PolynomialInterpolation qui permet
d'interpoler un ensemble de points en utilisant :
- la méthode de Lagrange
- la méthode de Newton avec différences divisées
"""

from __future__ import annotations

import numpy as np


class PolynomialInterpolation:
    """
    Classe d'interpolation polynomiale.

    Cette classe permet de construire un polynôme interpolateur à partir
    de points expérimentaux, puis de l'évaluer avec la méthode de Lagrange
    ou la méthode de Newton.

    Attributes
    ----------
    x_points : np.ndarray
        Abscisses des points d'interpolation.
    y_points : np.ndarray
        Ordonnées des points d'interpolation.
    n : int
        Nombre de points.
    _newton_coeffs : np.ndarray | None
        Coefficients des différences divisées pour la forme de Newton.
    """

    def __init__(self, x_points, y_points):
        """
        Initialise l'interpolateur polynomial.

        Parameters
        ----------
        x_points : array-like
            Liste ou tableau des abscisses.
        y_points : array-like
            Liste ou tableau des ordonnées.

        Raises
        ------
        ValueError
            Si les tableaux ont des longueurs différentes.
            Si moins de deux points sont fournis.
            Si les points x ne sont pas distincts.
        """
        self.x_points = np.asarray(x_points, dtype=float)
        self.y_points = np.asarray(y_points, dtype=float)

        if self.x_points.ndim != 1 or self.y_points.ndim != 1:
            raise ValueError("x_points et y_points doivent être des tableaux 1D.")

        if len(self.x_points) != len(self.y_points):
            raise ValueError("x_points et y_points doivent avoir la même longueur.")

        if len(self.x_points) < 2:
            raise ValueError("Au moins deux points sont nécessaires.")

        if len(np.unique(self.x_points)) != len(self.x_points):
            raise ValueError("Les points x doivent être distincts.")

        self.n = len(self.x_points)
        self._newton_coeffs = None

    def lagrange(self, x_eval):
        """
        Évalue le polynôme interpolateur par la formule de Lagrange.

        Parameters
        ----------
        x_eval : float or array-like
            Point ou ensemble de points où évaluer le polynôme.

        Returns
        -------
        float or np.ndarray
            Valeur(s) du polynôme interpolateur.
        """
        x_eval_array = np.atleast_1d(np.asarray(x_eval, dtype=float))
        result = np.zeros_like(x_eval_array, dtype=float)

        for i in range(self.n):
            basis = np.ones_like(x_eval_array, dtype=float)
            for j in range(self.n):
                if i != j:
                    basis *= (
                        (x_eval_array - self.x_points[j]) /
                        (self.x_points[i] - self.x_points[j])
                    )
            result += self.y_points[i] * basis

        if np.isscalar(x_eval):
            return float(result[0])
        return result

    def newton_coefficients(self):
        """
        Calcule les coefficients de Newton par différences divisées.

        Returns
        -------
        np.ndarray
            Tableau des coefficients de Newton.
        """
        coeffs = np.copy(self.y_points).astype(float)

        for j in range(1, self.n):
            coeffs[j:self.n] = (
                (coeffs[j:self.n] - coeffs[j - 1:self.n - 1]) /
                (self.x_points[j:self.n] - self.x_points[0:self.n - j])
            )

        self._newton_coeffs = coeffs
        return coeffs

    def newton_eval(self, x_eval):
        """
        Évalue le polynôme interpolateur sous forme de Newton.

        Parameters
        ----------
        x_eval : float or array-like
            Point ou ensemble de points où évaluer le polynôme.

        Returns
        -------
        float or np.ndarray
            Valeur(s) du polynôme interpolateur.
        """
        if self._newton_coeffs is None:
            self.newton_coefficients()

        x_eval_array = np.atleast_1d(np.asarray(x_eval, dtype=float))
        result = np.full_like(
            x_eval_array,
            self._newton_coeffs[-1],
            dtype=float
        )

        for i in range(self.n - 2, -1, -1):
            result = self._newton_coeffs[i] + (
                x_eval_array - self.x_points[i]
            ) * result

        if np.isscalar(x_eval):
            return float(result[0])
        return result

    def evaluate(self, x_eval, method="newton"):
        """
        Évalue le polynôme interpolateur avec la méthode choisie.

        Parameters
        ----------
        x_eval : float or array-like
            Point ou ensemble de points où évaluer le polynôme.
        method : str, optional
            Méthode à utiliser : 'newton' ou 'lagrange'.
            Par défaut : 'newton'.

        Returns
        -------
        float or np.ndarray
            Valeur(s) interpolées.

        Raises
        ------
        ValueError
            Si la méthode demandée n'est pas supportée.
        """
        method = method.lower()

        if method == "newton":
            return self.newton_eval(x_eval)
        if method == "lagrange":
            return self.lagrange(x_eval)

        raise ValueError(
            "Méthode invalide. Choisissez 'newton' ou 'lagrange'."
        )