#Algoritmo Polinômio-Lagrange
import numpy as np
import matplotlib.pyplot as plt

def lagrange(n_points, x_values: np.array, y_values: np.array, interpolation_value: float) -> float:
    interpolated_value: float = 0.0
    if interpolated_value == interpolation_value: return
    for i in range( n_points ):
        p: float = 1.0
        for j in range( n_points ):
            if i != j:
                p *= ( interpolation_value - x_values[j] ) / ( x_values[i] - x_values[j] )
        interpolated_value += y_values[i] * p

    return interpolated_value

def plot_lagrange_interpolation( x_values: np.array, y_values: np.array, num_points: int = 100 ) -> None:

    xplt = np.linspace( x_values[0], x_values[-1], num_points )
    yplt = np.zeros_like(xplt)

    for i, x in enumerate(xplt):
        yplt[i] = lagrange(len(x_values), x_values, y_values, x)

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, 'o', label='Pontos de Dados')
    plt.plot(xplt, yplt, '-', label='Interpolação de Lagrange')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolação de Lagrange')
    plt.legend()
    plt.grid(True)
    plt.show()

x_values = np.array( [0, 20, 40, 60, 80, 100], dtype=float )
y_values = np.array( [26, 48.6, 61.6, 71.2, 74.8, 75.2], dtype=float )
interpolation_value = 80
n = len(x_values)

plot_lagrange_interpolation(x_values, y_values, 100)