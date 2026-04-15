"""
Module d'intégration numérique par les méthodes de Newton-Cotes.

Ce module implémente la classe NewtonCotes proposant les méthodes
classiques d'intégration numérique composée : rectangle, trapèzes,
Simpson 1/3 et Simpson 3/8 (bonus).
"""

import numpy as np


class NewtonCotes:
    """
    Classe regroupant les méthodes de quadrature de Newton-Cotes.

    Toutes les méthodes sont statiques : aucune instanciation n'est
    nécessaire.  Chaque méthode applique la version *composée* de la
    règle correspondante, ce qui permet de contrôler la précision via
    le paramètre ``n`` (nombre de sous-intervalles).

    Exemple d'utilisation::

        from newton_cotes import NewtonCotes
        import numpy as np

        result = NewtonCotes.simpson(np.exp, 0, 1, n=100)
        print(result)   # ≈ 1.7182818284590453
    """

    # ------------------------------------------------------------------
    # Méthode du rectangle (point milieu composée)
    # ------------------------------------------------------------------

    @staticmethod
    def rectangle(f, a, b, n=1):
        """
        Intègre *f* sur [a, b] par la règle du rectangle composée
        (point milieu).

        La règle simple évalue ``f`` au centre de chaque sous-intervalle
        et multiplie par la largeur ``h`` :

        .. math::

            \\int_a^b f(x)\\,dx \\approx h \\sum_{i=0}^{n-1}
            f\\!\\left(a + \\left(i + \\tfrac{1}{2}\\right) h\\right),
            \\quad h = \\frac{b - a}{n}

        L'erreur globale est en :math:`O(h^2)` (ordre 2).

        Parameters
        ----------
        f : callable
            Fonction à intégrer.  Doit accepter un scalaire ou un
            tableau NumPy et renvoyer une valeur de même forme.
        a : float
            Borne inférieure d'intégration.
        b : float
            Borne supérieure d'intégration.
        n : int, optional
            Nombre de sous-intervalles (par défaut 1).  Doit être ≥ 1.

        Returns
        -------
        float
            Valeur approchée de l'intégrale.

        Raises
        ------
        ValueError
            Si ``n`` est inférieur à 1 ou si ``a >= b``.

        Examples
        --------
        >>> import numpy as np
        >>> NewtonCotes.rectangle(np.exp, 0, 1, n=1000)
        1.7182818284...
        """
        if n < 1:
            raise ValueError(
                f"Le nombre de sous-intervalles n doit être ≥ 1, reçu n={n}."
            )
        if a >= b:
            raise ValueError(
                f"La borne inférieure a={a} doit être strictement "
                f"inférieure à la borne supérieure b={b}."
            )

        h = (b - a) / n
        # Points milieu de chaque sous-intervalle
        midpoints = a + (np.arange(n) + 0.5) * h
        return h * np.sum(f(midpoints))

    # ------------------------------------------------------------------
    # Méthode des trapèzes composée
    # ------------------------------------------------------------------

    @staticmethod
    def trapezoidal(f, a, b, n=1):
        """
        Intègre *f* sur [a, b] par la règle des trapèzes composée.

        .. math::

            \\int_a^b f(x)\\,dx \\approx
            \\frac{h}{2}\\!\\left[f(a) + 2\\sum_{i=1}^{n-1} f(x_i)
            + f(b)\\right],
            \\quad h = \\frac{b - a}{n}

        L'erreur globale est en :math:`O(h^2)` (ordre 2).

        Parameters
        ----------
        f : callable
            Fonction à intégrer.
        a : float
            Borne inférieure d'intégration.
        b : float
            Borne supérieure d'intégration.
        n : int, optional
            Nombre de sous-intervalles (par défaut 1).  Doit être ≥ 1.

        Returns
        -------
        float
            Valeur approchée de l'intégrale.

        Raises
        ------
        ValueError
            Si ``n`` est inférieur à 1 ou si ``a >= b``.

        Examples
        --------
        >>> import numpy as np
        >>> NewtonCotes.trapezoidal(np.exp, 0, 1, n=1000)
        1.7182818284...
        """
        if n < 1:
            raise ValueError(
                f"Le nombre de sous-intervalles n doit être ≥ 1, reçu n={n}."
            )
        if a >= b:
            raise ValueError(
                f"La borne inférieure a={a} doit être strictement "
                f"inférieure à la borne supérieure b={b}."
            )

        h = (b - a) / n
        x = a + np.arange(n + 1) * h  # n+1 nœuds
        y = f(x)
        # Coefficients : 1, 2, 2, ..., 2, 1
        return h / 2 * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

    # ------------------------------------------------------------------
    # Méthode de Simpson 1/3 composée
    # ------------------------------------------------------------------

    @staticmethod
    def simpson(f, a, b, n=2):
        """
        Intègre *f* sur [a, b] par la règle de Simpson 1/3 composée.

        .. math::

            \\int_a^b f(x)\\,dx \\approx
            \\frac{h}{3}\\!\\left[f(x_0) +
            4\\sum_{\\substack{i=1 \\\\ i \\text{ impair}}}^{n-1} f(x_i) +
            2\\sum_{\\substack{i=2 \\\\ i \\text{ pair}}}^{n-2} f(x_i) +
            f(x_n)\\right]

        avec :math:`h = (b - a) / n`.  L'erreur globale est en
        :math:`O(h^4)` (ordre 4).

        Parameters
        ----------
        f : callable
            Fonction à intégrer.
        a : float
            Borne inférieure d'intégration.
        b : float
            Borne supérieure d'intégration.
        n : int, optional
            Nombre de sous-intervalles (par défaut 2).
            **Doit être pair et ≥ 2.**

        Returns
        -------
        float
            Valeur approchée de l'intégrale.

        Raises
        ------
        ValueError
            Si ``n`` est impair, inférieur à 2, ou si ``a >= b``.

        Examples
        --------
        >>> import numpy as np
        >>> NewtonCotes.simpson(np.exp, 0, 1, n=100)
        1.7182818284590453
        """
        if n < 2:
            raise ValueError(
                f"Le nombre de sous-intervalles n doit être ≥ 2, reçu n={n}."
            )
        if n % 2 != 0:
            raise ValueError(
                f"La méthode de Simpson 1/3 exige un nombre pair de "
                f"sous-intervalles, reçu n={n} (impair).  "
                f"Essayez n={n + 1}."
            )
        if a >= b:
            raise ValueError(
                f"La borne inférieure a={a} doit être strictement "
                f"inférieure à la borne supérieure b={b}."
            )

        h = (b - a) / n
        x = a + np.arange(n + 1) * h
        y = f(x)

        # Coefficients : 1, 4, 2, 4, 2, ..., 4, 1
        coeffs = np.ones(n + 1)
        coeffs[1:-1:2] = 4   # indices impairs
        coeffs[2:-2:2] = 2   # indices pairs (hors extrémités)

        return h / 3 * np.dot(coeffs, y)

    # ------------------------------------------------------------------
    # Méthode de Simpson 3/8 composée  (BONUS)
    # ------------------------------------------------------------------

    @staticmethod
    def simpson_38(f, a, b, n=3):
        """
        Intègre *f* sur [a, b] par la règle de Simpson 3/8 composée
        (méthode bonus).

        La règle simple de Simpson 3/8 utilise 4 nœuds équidistants :

        .. math::

            \\int_a^b f(x)\\,dx \\approx
            \\frac{3h}{8}\\bigl[f(x_0) + 3f(x_1) + 3f(x_2) + f(x_3)\\bigr]

        La version composée décompose [a, b] en groupes de 3
        sous-intervalles et applique la règle simple sur chaque groupe.
        **n doit donc être un multiple de 3.**

        L'erreur globale est en :math:`O(h^4)` (ordre 4), identique à
        Simpson 1/3.

        Parameters
        ----------
        f : callable
            Fonction à intégrer.
        a : float
            Borne inférieure d'intégration.
        b : float
            Borne supérieure d'intégration.
        n : int, optional
            Nombre de sous-intervalles (par défaut 3).
            **Doit être un multiple de 3 et ≥ 3.**

        Returns
        -------
        float
            Valeur approchée de l'intégrale.

        Raises
        ------
        ValueError
            Si ``n`` n'est pas un multiple de 3, inférieur à 3, ou si
            ``a >= b``.

        Examples
        --------
        >>> import numpy as np
        >>> NewtonCotes.simpson_38(np.exp, 0, 1, n=99)
        1.7182818284590453
        """
        if n < 3:
            raise ValueError(
                f"Le nombre de sous-intervalles n doit être ≥ 3, reçu n={n}."
            )
        if n % 3 != 0:
            raise ValueError(
                f"La méthode de Simpson 3/8 exige que n soit un multiple "
                f"de 3, reçu n={n}.  Essayez n={n + (3 - n % 3)}."
            )
        if a >= b:
            raise ValueError(
                f"La borne inférieure a={a} doit être strictement "
                f"inférieure à la borne supérieure b={b}."
            )

        h = (b - a) / n
        x = a + np.arange(n + 1) * h
        y = f(x)

        # Coefficients : 1, 3, 3, 2, 3, 3, 2, ..., 3, 3, 1
        coeffs = np.ones(n + 1)
        # Tous les nœuds intérieurs commencent à 3
        coeffs[1:-1] = 3
        # Les nœuds aux jonctions de groupes (multiples de 3, sauf
        # les extrémités) reçoivent le coefficient 2
        coeffs[3:-1:3] = 2

        return 3 * h / 8 * np.dot(coeffs, y)