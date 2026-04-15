from __future__ import annotations

import numpy as np

from src.interpolation.polynomial_interpolation import PolynomialInterpolation


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
        t0 = self.t_data[0]

        result = self.T_ambient + (T0 - self.T_ambient) * np.exp(
            -k * (t_eval_array - t0)
        )

        if np.isscalar(t_eval):
            return float(result[0])
        return result