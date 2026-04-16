from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from src.interpolation.polynomial_interpolation import PolynomialInterpolation
from src.integration.adaptive import AdaptiveIntegration
from src.integration.newton_cotes import NewtonCotes

class CoolingProblem:
    """
    Modélise le problème de refroidissement d’un composant électronique
    à partir de données expérimentales.
    """

    def __init__(self, t_data, T_data, T_ambient=20.0, h_coeff=50.0):
        """
        Initialise le problème de refroidissement.

        Parameters
        ----------
        t_data : array-like
            Instants de mesure.
        T_data : array-like
            Températures mesurées.
        T_ambient : float, optional
            Température ambiante en °C. Par défaut : 20.0.
        h_coeff : float, optional
            Coefficient de transfert thermique. Par défaut : 50.0.
        """
        self.t_data = np.asarray(t_data, dtype=float)
        self.T_data = np.asarray(T_data, dtype=float)
        self.T_ambient = float(T_ambient)
        self.h_coeff = float(h_coeff)

        if self.h_coeff < 0:
            raise ValueError("h_coeff doit être positif.")

        self.interpolator = PolynomialInterpolation(self.t_data, self.T_data)


    def temperature(self, t_eval):
        """
        Évalue la température interpolée à l’instant demandé.

        Parameters
        ----------
        t_eval : float or array-like
        Instant(s) où évaluer la température.

        Returns
        -------
        float or np.ndarray
            Température(s) interpolée(s).
        """
        return self.interpolator.evaluate(t_eval, method="newton")
    
    def heat_loss_rate(self, t_eval):
        """
        Calcule le taux de perte de chaleur à l’instant demandé.

        Parameters
        ----------
        t_eval : float or array-like
            Instant(s) où évaluer le taux de perte thermique.

        Returns
        -------
        float or np.ndarray
            Taux de perte de chaleur.
        """
        return self.h_coeff * (self.temperature(t_eval) - self.T_ambient)
    
    def total_heat_loss(self, method="adaptive", n=100):
        """
        Calcule la quantité totale de chaleur dissipée sur l’intervalle
        des données expérimentales.

        Parameters
        ----------
        method : str, optional
            Méthode d’intégration à utiliser :
            'rectangle', 'trapezoidal', 'simpson' ou 'adaptive'.
            Par défaut : 'adaptive'.
        n : int, optional
            Nombre de sous-intervalles pour les méthodes composées.
            Par défaut : 100.

        Returns
        -------
        float
            Quantité totale de chaleur dissipée.

        Raises
        ------
        ValueError
            Si la méthode demandée n'est pas supportée.
        """
        a = self.t_data[0]
        b = self.t_data[-1]

        def integrand(t):
            return self.heat_loss_rate(t)

        method = method.lower()

        if method == "rectangle":
            return NewtonCotes.rectangle(integrand, a, b, n=n)

        if method == "trapezoidal":
            return NewtonCotes.trapezoidal(integrand, a, b, n=n)

        if method == "simpson":
            return NewtonCotes.simpson(integrand, a, b, n=n)

        if method == "adaptive":
            integrator = AdaptiveIntegration()
            return integrator.adaptive_simpson(integrand, a, b)

        raise ValueError(
            "Méthode invalide. Choisissez "
            "'rectangle', 'trapezoidal', 'simpson' ou 'adaptive'."
        )
    
    def exponential_model(self, t_eval, k):
        """
        Évalue le modèle exponentiel de refroidissement.

        Parameters
        ----------
        t_eval : float or array-like
            Instant(s) où évaluer le modèle.
        k : float
            Constante de refroidissement.

        Returns
        -------
        float or np.ndarray
            Température(s) donnée(s) par le modèle exponentiel.

        Raises
        ------
        ValueError
            Si k est négatif.
        """
        if k < 0:
            raise ValueError("k doit être positif ou nul.")

        t_eval_array = np.atleast_1d(np.asarray(t_eval, dtype=float))
        T0 = self.T_data[0]

        result = self.T_ambient + (T0 - self.T_ambient) * np.exp(-k * t_eval_array)

        if np.isscalar(t_eval):
            return float(result[0])
        return result

        
    def model_error(self, k):
        """
        Calcule l'erreur entre la température interpolée et le modèle
        exponentiel pour une valeur donnée de k.

        Parameters
        ----------
        k : float
            Constante de refroidissement.

        Returns
        -------
        float
            Erreur intégrée absolue entre l'interpolation et le modèle.

        Raises
        ------
        ValueError
            Si k est négatif.
        """
        if k < 0:
            raise ValueError("k doit être positif ou nul.")

        a = self.t_data[0]
        b = self.t_data[-1]

        def error_function(t):
            return np.abs(self.temperature(t) - self.exponential_model(t, k))

        integrator = AdaptiveIntegration()
        return integrator.adaptive_simpson(error_function, a, b)

    def estimate_k(self, k_min=0.01, k_max=0.5, tol=1e-4):
        """
        Estime la valeur optimale de k en minimisant l'erreur du modèle
        exponentiel sur l'intervalle [k_min, k_max].

        Parameters
        ----------
        k_min : float, optional
            Borne inférieure de recherche. Par défaut : 0.01.
        k_max : float, optional
            Borne supérieure de recherche. Par défaut : 0.5.
        tol : float, optional
            Tolérance d'arrêt. Par défaut : 1e-4.

        Returns
        -------
        float
            Valeur estimée optimale de k.

        Raises
        ------
        ValueError
            Si les bornes ou la tolérance sont invalides.
        """
        if k_min < 0 or k_max < 0:
            raise ValueError("k_min et k_max doivent être positifs ou nuls.")

        if k_min >= k_max:
            raise ValueError("Il faut avoir k_min < k_max.")

        if tol <= 0:
            raise ValueError("tol doit être strictement positive.")

        phi = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi

        a = k_min
        b = k_max

        c = a + resphi * (b - a)
        d = b - resphi * (b - a)

        fc = self.model_error(c)
        fd = self.model_error(d)

        while (b - a) > tol:
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = a + resphi * (b - a)
                fc = self.model_error(c)
            else:
                a = c
                c = d
                fc = fd
                d = b - resphi * (b - a)
                fd = self.model_error(d)

        return (a + b) / 2