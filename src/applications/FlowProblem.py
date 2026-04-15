"""
Module modélisant le problème d'écoulement dans un canal.

Ce module implémente la classe FlowProblem qui permet d'analyser
le profil de vitesse d'un fluide dans un canal de largeur variable,
et d'en déduire le débit volumique, l'accélération et le travail.
"""

from __future__ import annotations

import numpy as np

from src.interpolation.polynomial_interpolation import PolynomialInterpolation
from src.integration.adaptive import AdaptiveIntegration
from src.integration.newton_cotes import NewtonCotes


class FlowProblem:
    """
    Modélise l'écoulement d'un fluide dans un canal à largeur variable.

    La vitesse est interpolée à partir de données expérimentales.
    La largeur du canal peut être fournie via une fonction personnalisée ;
    par défaut on utilise la fonction du sujet : w(x) = 0.5 + 0.1 * x.

    Attributs
    ---------
    x_data : np.ndarray
        Positions des mesures de vitesse.
    v_data : np.ndarray
        Vitesses mesurées aux positions x_data.
    width_func : callable
        Fonction w(x) donnant la largeur du canal en x.
    interpolator : PolynomialInterpolation
        Interpolateur polynomial construit sur les données de vitesse.

    Exemple d'utilisation
    ---------------------
    >>> x = [0, 1, 2, 3, 4, 5, 6]
    >>> v = [0.5, 1.2, 1.8, 2.1, 1.9, 1.4, 0.8]
    >>> fp = FlowProblem(x, v)
    >>> fp.velocity(3.0)
    2.1
    >>> fp.total_flow_rate()
    # débit volumique total sur [x_data[0], x_data[-1]]
    """

    def __init__(self, x_data, v_data, width_func=None):
        """
        Initialise le problème d'écoulement.

        Parameters
        ----------
        x_data : array-like
            Positions (en m) où la vitesse a été mesurée.
            Les valeurs doivent être distinctes et triées.
        v_data : array-like
            Vitesses mesurées (en m/s) aux positions x_data.
            Doit avoir la même longueur que x_data.
        width_func : callable or None, optional
            Fonction w(x) → largeur du canal (en m).
            Si None, on utilise w(x) = 0.5 + 0.1 * x (valeur du sujet).

        Raises
        ------
        ValueError
            Si x_data et v_data n'ont pas la même longueur,
            ou si x_data contient moins de 2 points.
        """
        self.x_data = np.asarray(x_data, dtype=float)
        self.v_data = np.asarray(v_data, dtype=float)

        if self.x_data.ndim != 1 or self.v_data.ndim != 1:
            raise ValueError("x_data et v_data doivent être des tableaux 1-D.")

        if len(self.x_data) != len(self.v_data):
            raise ValueError(
                f"x_data (longueur {len(self.x_data)}) et v_data "
                f"(longueur {len(self.v_data)}) doivent avoir la même longueur."
            )

        if len(self.x_data) < 2:
            raise ValueError(
                "Au moins 2 points de données sont nécessaires."
            )

        # Fonction de largeur par défaut : w(x) = 0.5 + 0.1·x  (sujet §1.1)
        if width_func is None:
            self.width_func = lambda x: 0.5 + 0.1 * np.asarray(x, dtype=float)
        else:
            self.width_func = width_func

        # Interpolateur polynomial (méthode de Newton par défaut)
        self.interpolator = PolynomialInterpolation(self.x_data, self.v_data)

        # Intégrateur adaptatif (utilisé par total_flow_rate)
        self._adaptive = AdaptiveIntegration()

    # ------------------------------------------------------------------
    # Vitesse interpolée
    # ------------------------------------------------------------------

    def velocity(self, x_eval):
        """
        Évalue la vitesse du fluide à la position x_eval par interpolation.

        Utilise la méthode de Newton (différences divisées) construite
        sur les données expérimentales.

        Parameters
        ----------
        x_eval : float or array-like
            Position(s) où évaluer la vitesse (en m).

        Returns
        -------
        float or np.ndarray
            Vitesse(s) interpolée(s) (en m/s).

        Examples
        --------
        >>> fp.velocity(0.0)   # doit renvoyer v_data[0]
        0.5
        """
        return self.interpolator.evaluate(x_eval, method="newton")

    # ------------------------------------------------------------------
    # Débit local
    # ------------------------------------------------------------------

    def local_flow_rate(self, x_eval):
        """
        Calcule le débit volumique local q(x) = v(x) · w(x).

        Le débit local est le produit de la vitesse interpolée par la
        largeur du canal en ce point.

        Parameters
        ----------
        x_eval : float or array-like
            Position(s) où évaluer le débit local (en m).

        Returns
        -------
        float or np.ndarray
            Débit(s) local(aux) (en m²/s).

        Examples
        --------
        >>> fp.local_flow_rate(0.0)
        # v(0) * w(0) = v_data[0] * 0.5
        """
        x_arr = np.asarray(x_eval, dtype=float)
        return self.velocity(x_arr) * self.width_func(x_arr)

    # ------------------------------------------------------------------
    # Débit total
    # ------------------------------------------------------------------

    def total_flow_rate(self, method="adaptive", n=100):
        """
        Calcule le débit volumique total D = ∫ v(x)·w(x) dx
        sur l'intervalle [x_data[0], x_data[-1]].

        Trois méthodes d'intégration sont disponibles :

        * ``'adaptive'``  : Simpson adaptatif (précision contrôlée).
        * ``'simpson'``   : Simpson composé (n doit être pair).
        * ``'trapezoidal'``: Trapèzes composés.

        Parameters
        ----------
        method : {'adaptive', 'simpson', 'trapezoidal'}, optional
            Méthode d'intégration numérique. Par défaut ``'adaptive'``.
        n : int, optional
            Nombre de sous-intervalles pour Simpson et Trapèzes.
            Ignoré si ``method='adaptive'``.  Par défaut 100.

        Returns
        -------
        float
            Débit volumique total (en m³/s).

        Raises
        ------
        ValueError
            Si la méthode demandée n'est pas reconnue.

        Examples
        --------
        >>> fp.total_flow_rate(method='adaptive')
        8.34...
        >>> fp.total_flow_rate(method='simpson', n=200)
        8.34...
        """
        a = float(self.x_data[0])
        b = float(self.x_data[-1])

        integrand = self.local_flow_rate  # q(x) = v(x) · w(x)

        if method == "adaptive":
            return self._adaptive.adaptive_simpson(integrand, a, b)

        if method == "simpson":
            # S'assurer que n est pair
            n_even = n if n % 2 == 0 else n + 1
            return NewtonCotes.simpson(integrand, a, b, n=n_even)

        if method == "trapezoidal":
            return NewtonCotes.trapezoidal(integrand, a, b, n=n)

        raise ValueError(
            f"Méthode d'intégration inconnue : '{method}'. "
            f"Choisir parmi 'adaptive', 'simpson', 'trapezoidal'."
        )

    # ------------------------------------------------------------------
    # Accélération
    # ------------------------------------------------------------------

    def acceleration(self, x_eval):
        """
        Estime l'accélération du fluide a(x) ≈ dv/dx par différence finie
        centrée d'ordre 2.

        .. math::

            a(x) \\approx \\frac{v(x + \\delta) - v(x - \\delta)}{2\\delta}

        avec :math:`\\delta = 10^{-5} \\cdot (x_{\\max} - x_{\\min})` pour
        être indépendant de l'échelle.

        Parameters
        ----------
        x_eval : float or array-like
            Position(s) où évaluer l'accélération (en m).

        Returns
        -------
        float or np.ndarray
            Accélération(s) (en m/s²).

        Notes
        -----
        L'accélération est ici spatiale (dv/dx), pas temporelle.

        Examples
        --------
        >>> fp.acceleration(3.0)
        # dérivée numérique de v en x=3
        """
        delta = 1e-5 * (self.x_data[-1] - self.x_data[0])
        x_arr = np.asarray(x_eval, dtype=float)
        return (self.velocity(x_arr + delta) - self.velocity(x_arr - delta)) / (
            2 * delta
        )

    # ------------------------------------------------------------------
    # Travail
    # ------------------------------------------------------------------

    def work(self, mass=2.0):
        """
        Calcule le travail exercé sur une particule de masse *mass*
        traversant le canal de x_data[0] à x_data[-1].

        Le travail est défini par :

        .. math::

            W = \\int_{x_0}^{x_f} F(x)\\,dx
            = \\int_{x_0}^{x_f} m \\cdot a(x)\\,dx
            = m \\cdot \\int_{x_0}^{x_f} \\frac{dv}{dx}\\,dx
            = m \\cdot \\bigl[v(x_f) - v(x_0)\\bigr]

        Ce résultat analytique (théorème fondamental du calcul) est
        utilisé directement, puis également vérifié par intégration
        numérique adaptive.

        Parameters
        ----------
        mass : float, optional
            Masse de la particule (en kg). Par défaut 2.0.

        Returns
        -------
        float
            Travail (en Joules).

        Raises
        ------
        ValueError
            Si mass est négatif ou nul.

        Examples
        --------
        >>> fp.work(mass=2.0)
        # 2 * (v(x_f) - v(x_0))
        """
        if mass <= 0:
            raise ValueError(
                f"La masse doit être strictement positive, reçu mass={mass}."
            )

        a = float(self.x_data[0])
        b = float(self.x_data[-1])

        # F(x) = m · a(x) = m · dv/dx
        force = lambda x: mass * self.acceleration(x)  # noqa: E731

        return self._adaptive.adaptive_simpson(force, a, b)