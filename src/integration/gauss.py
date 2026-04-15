"""
Gaussian quadrature methods.
"""

import math


class GaussQuadrature:
    """
    Class for Gaussian quadrature integration.
    """

    @staticmethod
    def gauss_legendre_2(f, a, b):
        """
        2-point Gauss-Legendre quadrature.

        Parameters
        ----------
        f : function
        a : float
        b : float

        Returns
        -------
        float
        """
        x1 = -1 / math.sqrt(3)
        x2 = 1 / math.sqrt(3)

        t1 = (b - a) / 2 * x1 + (a + b) / 2
        t2 = (b - a) / 2 * x2 + (a + b) / 2

        return (b - a) / 2 * (f(t1) + f(t2))

    @staticmethod
    def gauss_legendre_3(f, a, b):
        """
        3-point Gauss-Legendre quadrature.

        Parameters
        ----------
        f : function
        a : float
        b : float

        Returns
        -------
        float
        """
        x1 = -math.sqrt(3/5)
        x2 = 0
        x3 = math.sqrt(3/5)

        w1 = 5/9
        w2 = 8/9
        w3 = 5/9

        t1 = (b - a) / 2 * x1 + (a + b) / 2
        t2 = (b - a) / 2 * x2 + (a + b) / 2
        t3 = (b - a) / 2 * x3 + (a + b) / 2

        return (b - a) / 2 * (
            w1 * f(t1) + w2 * f(t2) + w3 * f(t3)
        )
