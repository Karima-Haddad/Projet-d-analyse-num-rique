from src.interpolation.polynomial_interpolation import PolynomialInterpolation

x_points = [0, 1, 2]
y_points = [1, 3, 2]

interp = PolynomialInterpolation(x_points, y_points)

print("Lagrange en 1.5 :", interp.lagrange(1.5))
print("Coefficients de Newton :", interp.newton_coefficients())
print("Newton en 1.5 :", interp.newton_eval(1.5))
print("Evaluate en 1.5 :", interp.evaluate(1.5, method="newton"))