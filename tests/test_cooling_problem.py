from src.applications.cooling_problem import CoolingProblem

t_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
T_data = [90, 85, 72, 63, 58, 52, 48, 45, 43, 41, 40]

cooling = CoolingProblem(t_data, T_data)

print("Température à t = 2.5 s :", cooling.temperature(2.5))
print("Taux de perte thermique à t = 2.5 s :", cooling.heat_loss_rate(2.5))
# print("Chaleur totale dissipée :", cooling.total_heat_loss())
# print("Erreur pour k = 0.1 :", cooling.model_error(0.1))
# print("Valeur optimale de k :", cooling.estimate_k())