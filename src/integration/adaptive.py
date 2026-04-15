"""
Adaptive numerical integration methods.
"""

class AdaptiveIntegration:
    """
    Class for adaptive numerical integration using Simpson's method.
    """

    def __init__(self, tol=1e-6, max_depth=20):
        """
        Initialize the adaptive integrator.

        Parameters
        ----------
        tol : float
            Tolerance for stopping criterion.
        max_depth : int
            Maximum recursion depth.
        """
        self.tol = tol
        self.max_depth = max_depth

    def simpson_rule(self, f, a, b):
        """
        Compute Simpson's rule on interval [a, b].

        Parameters
        ----------
        f : function
            Function to integrate.
        a : float
            Lower bound.
        b : float
            Upper bound.

        Returns
        -------
        float
            Approximation of the integral.
        """
        c = (a + b) / 2
        return (b - a) / 6 * (f(a) + 4 * f(c) + f(b))

    def adaptive_simpson(self, f, a, b, tol=None, depth=0):
        """
        Recursive adaptive Simpson method.

        Parameters
        ----------
        f : function
        a : float
        b : float
        tol : float
        depth : int

        Returns
        -------
        float
        """
        if tol is None:
            tol = self.tol

        c = (a + b) / 2

        left = self.simpson_rule(f, a, c)
        right = self.simpson_rule(f, c, b)
        whole = self.simpson_rule(f, a, b)

        if depth >= self.max_depth:
            return left + right

        if abs(left + right - whole) < 15 * tol:
            return left + right + (left + right - whole) / 15

        return (
            self.adaptive_simpson(f, a, c, tol / 2, depth + 1) +
            self.adaptive_simpson(f, c, b, tol / 2, depth + 1)
        )
