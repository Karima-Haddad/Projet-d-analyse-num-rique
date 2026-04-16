from src.applications.cooling_problem import CoolingProblem

t_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
T_data = [90, 85, 72, 63, 58, 52, 48, 45, 43, 41, 40]

cooling = CoolingProblem(t_data, T_data)

print("Température à t = 2.5 :", cooling.temperature(2.5))
print("Température à t = 7.3 :", cooling.temperature(7.3))

print("Taux de perte à t = 2.5 :", cooling.heat_loss_rate(2.5))
print("Taux de perte à t = 7.3 :", cooling.heat_loss_rate(7.3))

print("Chaleur totale (rectangle) :", cooling.total_heat_loss(method="rectangle", n=100))
print("Chaleur totale (trapèzes) :", cooling.total_heat_loss(method="trapezoidal", n=100))
print("Chaleur totale (Simpson) :", cooling.total_heat_loss(method="simpson", n=100))
print("Chaleur totale (adaptive) :", cooling.total_heat_loss(method="adaptive"))

print("Modèle exponentiel à t = 2.5 avec k = 0.1 :", cooling.exponential_model(2.5, 0.1))
print("Erreur du modèle pour k = 0.1 :", cooling.model_error(0.1))

k_opt = cooling.estimate_k()
print("k optimal :", k_opt)
print("Erreur minimale :", cooling.model_error(k_opt))