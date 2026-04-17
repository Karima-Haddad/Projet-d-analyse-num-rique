from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =========================
# IMPORTS DE TON PROJET
# =========================
# Adapte ces imports selon l'organisation réelle de ton projet.

from src.interpolation.polynomial_interpolation import PolynomialInterpolation
from src.integration.newton_cotes import NewtonCotes
from src.integration.adaptive import AdaptiveIntegration

# Bonus si tu les as
try:
    from src.integration.gauss import GaussQuadrature
except ImportError:
    GaussQuadrature = None

from src.applications.cooling_problem import CoolingProblem
from src.applications.FlowProblem import FlowProblem
from src.visualization.visualizer import Visualizer


# =========================
# DOSSIERS DE SORTIE
# =========================
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = RESULTS_DIR / "data"


def ensure_directories() -> None:
    """Crée les dossiers de sortie s'ils n'existent pas."""
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


# =========================
# DONNÉES EXPÉRIMENTALES
# =========================
def load_experimental_data():
    """
    Charge les données du projet.

    Returns
    -------
    dict
        Dictionnaire contenant les données de refroidissement et d'écoulement.
    """
    cooling_data = {
        "t": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float),
        "T": np.array([90, 85, 72, 63, 58, 52, 48, 45, 43, 41, 40], dtype=float),
        "T_ambient": 20.0,
        "h_coeff": 50.0,
    }

    flow_data = {
        "x": np.array([0, 0.5, 1.2, 1.8, 2.5, 3.1, 3.7, 4.2, 4.8, 5.3, 6.0], dtype=float),
        "v": np.array([0, 2.1, 3.8, 5.2, 6.4, 7.0, 7.3, 7.2, 6.8, 5.9, 4.5], dtype=float),
    }

    return {"cooling": cooling_data, "flow": flow_data}


# =========================
# OUTILS
# =========================
def save_json(data: dict, filepath: Path) -> None:
    """Sauvegarde un dictionnaire en JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_current_figure(filename: str) -> None:
    """Sauvegarde la figure matplotlib courante."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def runge_function(x):
    """Fonction classique de Runge."""
    return 1.0 / (1.0 + 25.0 * x**2)


def chebyshev_nodes(a: float, b: float, n: int) -> np.ndarray:
    """
    Génère n points de Tchebychev sur [a, b].
    """
    k = np.arange(1, n + 1)
    x_cheb = np.cos((2 * k - 1) * np.pi / (2 * n))
    return (a + b) / 2 + (b - a) / 2 * x_cheb


def test_function(x):
    """Fonction test pour la convergence."""
    return np.exp(x)


def exact_integral_exp_0_1():
    """Valeur exacte de ∫_0^1 e^x dx."""
    return np.e - 1.0


# =========================
# 1) PHÉNOMÈNE DE RUNGE
# =========================
def study_runge(visualizer: Visualizer) -> dict:
    """
    Étudie le phénomène de Runge pour plusieurs valeurs de n,
    avec points équidistants et points de Tchebychev.
    """
    print("\n" + "=" * 60)
    print("ÉTUDE DU PHÉNOMÈNE DE RUNGE")
    print("=" * 60)

    a, b = -1.0, 1.0
    x_fine = np.linspace(a, b, 1000)
    y_true = runge_function(x_fine)

    degrees = [5, 10, 15, 20]
    results = {}

    for n in degrees:
        x_equi = np.linspace(a, b, n + 1)
        y_equi = runge_function(x_equi)

        x_cheb = chebyshev_nodes(a, b, n + 1)
        y_cheb = runge_function(x_cheb)

        interp_equi = PolynomialInterpolation(x_equi, y_equi)
        interp_cheb = PolynomialInterpolation(x_cheb, y_cheb)

        y_equi_interp = interp_equi.evaluate(x_fine, method="newton")
        y_cheb_interp = interp_cheb.evaluate(x_fine, method="newton")

        err_equi = float(np.max(np.abs(y_true - y_equi_interp)))
        err_cheb = float(np.max(np.abs(y_true - y_cheb_interp)))

        results[f"n_{n}"] = {
            "max_error_equidistant": err_equi,
            "max_error_chebyshev": err_cheb,
        }

        print(f"\nDegré n = {n}")
        print(f"Erreur max (points équidistants) : {err_equi:.6e}")
        print(f"Erreur max (points de Tchebychev) : {err_cheb:.6e}")

        interpolations = {
            "Équidistants": y_equi_interp,
            "Tchebychev": y_cheb_interp,
        }

        visualizer.plot_runge_phenomenon(x_fine, y_true, interpolations)
        plt.scatter(x_equi, y_equi, label="Nœuds équidistants", marker="o")
        plt.scatter(x_cheb, y_cheb, label="Nœuds Tchebychev", marker="x")
        plt.title(f"Phénomène de Runge (n = {n})")
        save_current_figure(f"runge_n_{n}.png")

    return results


# =========================
# 2) REFROIDISSEMENT
# =========================
def analyze_cooling(data: dict, visualizer: Visualizer) -> dict:
    """
    Analyse le problème de refroidissement :
    interpolation, pertes de chaleur, estimation de k.
    """
    print("\n" + "=" * 60)
    print("ANALYSE DU PROBLÈME DE REFROIDISSEMENT")
    print("=" * 60)

    t_data = data["t"]
    T_data = data["T"]
    T_ambient = data["T_ambient"]
    h_coeff = data["h_coeff"]

    cooling = CoolingProblem(
        t_data=t_data,
        T_data=T_data,
        T_ambient=T_ambient,
        h_coeff=h_coeff,
    )

    t_fine = np.linspace(t_data.min(), t_data.max(), 500)
    T_interp = cooling.temperature(t_fine)

    # COMPARAISON LAGRANGE vs NEWTON
    interp = PolynomialInterpolation(t_data, T_data)

    T_lagrange = interp.evaluate(t_fine, method="lagrange")
    T_newton = interp.evaluate(t_fine, method="newton")

    plt.figure(figsize=(10, 6))
    plt.scatter(t_data, T_data, label="Données expérimentales", color="red", zorder=5)

    plt.plot(t_fine, T_lagrange, label="Lagrange")
    plt.plot(t_fine, T_newton, linestyle="--", label="Newton", linewidth=2)

    plt.xlabel("Temps (s)")
    plt.ylabel("Température (°C)")
    plt.title("Comparaison Lagrange vs Newton")
    plt.legend()
    plt.grid(True)

    save_current_figure("lagrange_vs_newton.png")

    # TEMPÉRATURES À t = 2.5 s ET t = 7.3 s
    t_eval_1 = 2.5
    t_eval_2 = 7.3

    T_25_lagrange = float(interp.evaluate(t_eval_1, method="lagrange"))
    T_25_newton = float(interp.evaluate(t_eval_1, method="newton"))

    T_73_lagrange = float(interp.evaluate(t_eval_2, method="lagrange"))
    T_73_newton = float(interp.evaluate(t_eval_2, method="newton"))

    print(f"T(2.5 s) par Lagrange = {T_25_lagrange:.4f} °C")
    print(f"T(2.5 s) par Newton   = {T_25_newton:.4f} °C")
    print(f"T(7.3 s) par Lagrange = {T_73_lagrange:.4f} °C")
    print(f"T(7.3 s) par Newton   = {T_73_newton:.4f} °C")

    # Questions du sujet
    t_eval_1 = 2.5
    t_eval_2 = 7.3
    T_25 = float(cooling.temperature(t_eval_1))
    T_73 = float(cooling.temperature(t_eval_2))

    print(f"T(2.5 s) ≈ {T_25:.4f} °C")
    print(f"T(7.3 s) ≈ {T_73:.4f} °C")

    # Quantité totale de chaleur dissipée
    Q_rect = float(cooling.total_heat_loss(method="rectangle", n=100))
    Q_trap = float(cooling.total_heat_loss(method="trapezoidal", n=100))
    Q_simp = float(cooling.total_heat_loss(method="simpson", n=100))
    Q_adapt = float(cooling.total_heat_loss(method="adaptive", n=100))

    print(f"Chaleur dissipée (rectangle)   : {Q_rect:.6f} J")
    print(f"Chaleur dissipée (trapèzes)    : {Q_trap:.6f} J")
    print(f"Chaleur dissipée (Simpson)     : {Q_simp:.6f} J")
    print(f"Chaleur dissipée (adaptative)  : {Q_adapt:.6f} J")

    # Estimation du paramètre k
    k_opt = float(cooling.estimate_k(k_min=0.01, k_max=0.5, tol=1e-4))
    T_model = cooling.exponential_model(t_fine, k_opt)

    print(f"k optimal ≈ {k_opt:.6f}")

    # Erreurs du modèle exponentiel
    T_model_data = cooling.exponential_model(t_data, k_opt)
    abs_errors = np.abs(T_data - T_model_data)
    max_error = float(np.max(abs_errors))
    mean_error = float(np.mean(abs_errors))

    print(f"Erreur maximale du modèle : {max_error:.6f} °C")
    print(f"Erreur moyenne du modèle  : {mean_error:.6f} °C")

    # Courbe E(k)
    k_values = np.linspace(0.01, 0.5, 200)
    E_values = [float(cooling.model_error(k)) for k in k_values]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, E_values, label="E(k)")
    plt.axvline(k_opt, linestyle="--", label=f"k optimal = {k_opt:.4f}")
    plt.xlabel("k")
    plt.ylabel("E(k)")
    plt.title("Erreur du modèle exponentiel en fonction de k")
    plt.legend()
    plt.grid(True)
    save_current_figure("cooling_error_vs_k.png")

    # Graphe principal du refroidissement
    visualizer.plot_cooling_analysis(t_data, T_data, t_fine, T_interp, k_opt, T_model)
    save_current_figure("cooling_analysis.png")

    return {
        "T_2.5": T_25,
        "T_7.3": T_73,
        "heat_loss_rectangle": Q_rect,
        "heat_loss_trapezoidal": Q_trap,
        "heat_loss_simpson": Q_simp,
        "heat_loss_adaptive": Q_adapt,
        "k_opt": k_opt,
        "max_error_model": max_error,
        "mean_error_model": mean_error,
    }


# =========================
# 3) ÉCOULEMENT
# =========================
def analyze_flow(data: dict, visualizer: Visualizer) -> dict:
    """
    Analyse le problème d'écoulement :
    vitesse interpolée, débit local, débit total, travail.
    """
    print("\n" + "=" * 60)
    print("ANALYSE DU PROBLÈME D'ÉCOULEMENT")
    print("=" * 60)

    x_data = data["x"]
    v_data = data["v"]

    def width_func(x):
        return 0.5 + 0.1 * np.asarray(x)

    flow = FlowProblem(x_data=x_data, v_data=v_data, width_func=width_func)

    x_fine = np.linspace(x_data.min(), x_data.max(), 500)
    v_interp = flow.velocity(x_fine)

    # Débit total selon plusieurs méthodes
    D_rect = float(flow.total_flow_rate(method="rectangle", n=100))
    D_trap = float(flow.total_flow_rate(method="trapezoidal", n=100))
    D_simp = float(flow.total_flow_rate(method="simpson", n=100))
    D_adapt = float(flow.total_flow_rate(method="adaptive", n=100))

    print(f"Débit total (rectangle)   : {D_rect:.6f}")
    print(f"Débit total (trapèzes)    : {D_trap:.6f}")
    print(f"Débit total (Simpson)     : {D_simp:.6f}")
    print(f"Débit total (adaptative)  : {D_adapt:.6f}")

    # Exemples complémentaires
    q_local_mid = float(flow.local_flow_rate(3.0))
    acc_mid = float(flow.acceleration(3.0))
    work_val = float(flow.work(mass=2.0))

    print(f"Débit local à x = 3.0 m      : {q_local_mid:.6f}")
    print(f"Accélération à x = 3.0 m     : {acc_mid:.6f}")
    print(f"Travail estimé (m=2.0 kg)    : {work_val:.6f}")

    visualizer.plot_flow_analysis(x_data, v_data, x_fine, v_interp, width_func)
    save_current_figure("flow_analysis.png")

    return {
        "flow_rate_rectangle": D_rect,
        "flow_rate_trapezoidal": D_trap,
        "flow_rate_simpson": D_simp,
        "flow_rate_adaptive": D_adapt,
        "local_flow_rate_x_3": q_local_mid,
        "acceleration_x_3": acc_mid,
        "work_mass_2": work_val,
    }


# =========================
# 4) CONVERGENCE INTÉGRATION
# =========================
def integration_error(method_name: str, n: int) -> float:
    """
    Calcule l'erreur d'intégration pour ∫_0^1 e^x dx.
    """
    exact = exact_integral_exp_0_1()

    if method_name == "rectangle":
        approx = NewtonCotes.rectangle(test_function, 0.0, 1.0, n=n)
    elif method_name == "trapezoidal":
        approx = NewtonCotes.trapezoidal(test_function, 0.0, 1.0, n=n)
    elif method_name == "simpson":
        if n % 2 != 0:
            n += 1
        approx = NewtonCotes.simpson(test_function, 0.0, 1.0, n=n)
    else:
        raise ValueError(f"Méthode inconnue : {method_name}")

    return abs(float(approx) - exact)


def study_integration_convergence(visualizer: Visualizer) -> dict:
    """
    Étudie la convergence des méthodes rectangle, trapèzes et Simpson.
    """
    print("\n" + "=" * 60)
    print("ÉTUDE DE LA CONVERGENCE DES MÉTHODES D'INTÉGRATION")
    print("=" * 60)

    n_values = np.array([2, 4, 8, 16, 32, 64, 128, 256], dtype=int)

    methods = ["rectangle", "trapezoidal", "simpson"]
    errors = {}

    for method in methods:
        method_errors = []
        print(f"\nMéthode : {method}")

        for n in n_values:
            err = integration_error(method, int(n))
            method_errors.append(err)
            print(f"n = {n:3d}  -> erreur = {err:.6e}")

        errors[method] = method_errors

    visualizer.plot_convergence(
        n_values=n_values,
        errors=errors,
        methods=methods,
        title="Convergence des méthodes d'intégration"
    )
    save_current_figure("integration_convergence.png")

    # Comparaison Simpson composée vs adaptative
    adaptive = AdaptiveIntegration(tol=1e-6, max_depth=20)
    exact = exact_integral_exp_0_1()

    simpson_100 = NewtonCotes.simpson(test_function, 0.0, 1.0, n=100)
    adapt_val = adaptive.adaptive_simpson(test_function, 0.0, 1.0)

    err_simpson_100 = abs(float(simpson_100) - exact)
    err_adapt = abs(float(adapt_val) - exact)

    print("\nComparaison supplémentaire :")
    print(f"Simpson composée (n=100)  -> erreur = {err_simpson_100:.6e}")
    print(f"Simpson adaptative        -> erreur = {err_adapt:.6e}")

    results = {
        "n_values": n_values.tolist(),
        "errors": {k: [float(v) for v in vals] for k, vals in errors.items()},
        "simpson_100_error": float(err_simpson_100),
        "adaptive_error": float(err_adapt),
    }

    if GaussQuadrature is not None:
        gauss2 = float(GaussQuadrature.gauss_legendre_2(test_function, 0.0, 1.0))
        gauss3 = float(GaussQuadrature.gauss_legendre_3(test_function, 0.0, 1.0))
        results["gauss_legendre_2_error"] = abs(gauss2 - exact)
        results["gauss_legendre_3_error"] = abs(gauss3 - exact)

        print(f"Gauss-Legendre 2 points   -> erreur = {abs(gauss2 - exact):.6e}")
        print(f"Gauss-Legendre 3 points   -> erreur = {abs(gauss3 - exact):.6e}")

    return results


# =========================
# 5) SAUVEGARDE D'UN RÉSUMÉ TEXTE
# =========================
def save_summary_text(all_results: dict) -> None:
    """
    Sauvegarde un résumé texte lisible des résultats.
    """
    lines = []
    lines.append("RÉSULTATS DU PROJET D'ANALYSE NUMÉRIQUE")
    lines.append("=" * 50)
    lines.append("")

    lines.append("1. Étude du phénomène de Runge")
    for key, values in all_results["runge"].items():
        lines.append(
            f"{key} -> erreur max équidistants = {values['max_error_equidistant']:.6e}, "
            f"erreur max Tchebychev = {values['max_error_chebyshev']:.6e}"
        )
    lines.append("")

    lines.append("2. Refroidissement")
    cooling = all_results["cooling"]
    lines.append(f"T(2.5) = {cooling['T_2.5']:.6f} °C")
    lines.append(f"T(7.3) = {cooling['T_7.3']:.6f} °C")
    lines.append(f"Q rectangle   = {cooling['heat_loss_rectangle']:.6f}")
    lines.append(f"Q trapèzes    = {cooling['heat_loss_trapezoidal']:.6f}")
    lines.append(f"Q Simpson     = {cooling['heat_loss_simpson']:.6f}")
    lines.append(f"Q adaptative  = {cooling['heat_loss_adaptive']:.6f}")
    lines.append(f"k optimal     = {cooling['k_opt']:.6f}")
    lines.append(f"Erreur max    = {cooling['max_error_model']:.6f}")
    lines.append(f"Erreur moyenne= {cooling['mean_error_model']:.6f}")
    lines.append("")

    lines.append("3. Écoulement")
    flow = all_results["flow"]
    lines.append(f"D rectangle   = {flow['flow_rate_rectangle']:.6f}")
    lines.append(f"D trapèzes    = {flow['flow_rate_trapezoidal']:.6f}")
    lines.append(f"D Simpson     = {flow['flow_rate_simpson']:.6f}")
    lines.append(f"D adaptative  = {flow['flow_rate_adaptive']:.6f}")
    lines.append(f"Débit local x=3 = {flow['local_flow_rate_x_3']:.6f}")
    lines.append(f"Accélération x=3 = {flow['acceleration_x_3']:.6f}")
    lines.append(f"Travail (m=2) = {flow['work_mass_2']:.6f}")
    lines.append("")

    lines.append("4. Convergence intégration")
    convergence = all_results["convergence"]
    lines.append(f"Erreur Simpson n=100 = {convergence['simpson_100_error']:.6e}")
    lines.append(f"Erreur adaptive      = {convergence['adaptive_error']:.6e}")

    txt_path = RESULTS_DIR / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# MAIN
# =========================
def main():
    """Point d'entrée principal du projet."""
    ensure_directories()

    print("=" * 60)
    print("PROJET D'ANALYSE NUMÉRIQUE")
    print("Interpolation et Intégration Numérique")
    print("=" * 60)

    # Initialisation du visualiseur
    visualizer = Visualizer(style="seaborn-v0_8-darkgrid", figsize=(10, 6))

    # 1. Charger les données expérimentales
    data = load_experimental_data()

    # 2. Étudier le phénomène de Runge
    runge_results = study_runge(visualizer)

    # 3. Analyser le problème de refroidissement
    cooling_results = analyze_cooling(data["cooling"], visualizer)

    # 4. Analyser le problème d’écoulement
    flow_results = analyze_flow(data["flow"], visualizer)

    # 5. Étudier la convergence des méthodes d’intégration
    convergence_results = study_integration_convergence(visualizer)

    # 6. Sauvegarder les résultats
    all_results = {
        "runge": runge_results,
        "cooling": cooling_results,
        "flow": flow_results,
        "convergence": convergence_results,
    }

    save_json(all_results, DATA_DIR / "results.json")
    save_summary_text(all_results)

    # 7. Affichage final
    print("\n" + "=" * 60)
    print("EXÉCUTION TERMINÉE AVEC SUCCÈS")
    print("=" * 60)
    print(f"Résultats JSON  : {DATA_DIR / 'results.json'}")
    print(f"Résumé texte    : {RESULTS_DIR / 'summary.txt'}")
    print(f"Figures         : {FIGURES_DIR}")


if __name__ == "__main__":
    main()

