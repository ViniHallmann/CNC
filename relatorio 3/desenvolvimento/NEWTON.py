#Algoritmo de interpolação de Newton
import numpy as np
import matplotlib.pyplot as plt

def newton( n_points, x: np.array, y: np.array, interpolation_value: float ) -> float:
    dely = np.zeros( n_points )

    for i in range( n_points ): dely[i] = y[i]

    for i in range( 1, n_points ):
        for j in range( n_points - 1, i - 1, -1 ):
            dely[j] = ( ( dely[j] - dely[j - 1] ) / ( x[j] - x[j - i] ) )

    interpolated_value = dely[n_points - 1]
    for i in range( n_points - 2, -1, -1 ):
        interpolated_value = dely[i] + ( interpolation_value - x[i] ) * interpolated_value

    return interpolated_value

def plot_newton_interpolation(  newton_func, x_values: np.array, y_values: np.array, num_points: int = 100, especial_point: int = None) -> None:

    xplt = np.linspace( x_values[0], x_values[-1], num_points )
    yplt = np.zeros_like(xplt)

    for i, x in enumerate(xplt):
        yplt[i] = newton_func(len(x_values), x_values, y_values, x)

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, 'o', color = '#1837E7', label='Pontos', zorder = 5 )
    plt.plot(xplt, yplt, '-', color = '#37E718', label = 'Interpolação de newton')
    
    if especial_point is not None:
        y_especial = newton_func( len( x_values ), x_values, y_values, especial_point )
        plt.scatter( especial_point, y_especial, color = '#E71837', label = f'Ponto em x = {especial_point}', zorder = 5 )
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolação de Lagrange')
    plt.legend()
    plt.grid(True)
    plt.show()

x = np.array( [0,    20,   40,   60,   80,  100], dtype=float )
y = np.array( [26, 48.6, 61.6, 71.2, 74.8, 75.2], dtype=float )
interpolation_value: float = 10
n: int = len(x)

plot_newton_interpolation( newton, x, y, 100, interpolation_value)
result = newton(n, x, y, interpolation_value)
print(f'O valor interpolado para x = {interpolation_value} é {result}')

