#Algoritmo de Aproximação Linear - Chapra
import numpy as np
import matplotlib.pyplot as plt
def chapra( n: int, x: np.array, y: np.array):
    sumx: float = 0 
    sumy: float = 0 
    sumxy: float = 0
    sumx2: float = 0
    st: float = 0 
    sr: float = 0

    for i in range(n):
        sumx += x[i]
        sumy += y[i]
        sumxy += x[i] * y[i]
        sumx2 += x[i]**2
    
    xm = sumx / n
    ym = sumy / n
    d = (n * sumx2 - sumx**2)

    if d != 0:
        a1 = (n * sumxy - sumx * sumy) / d
    else:
        a1 = 0

    a0 = ym - a1*xm
    for i in range(n):
        st += (y[i] - ym)**2
        sr += (y[i] - a1*x[i] - a0 )**2
    if n > 2:
        syx = (sr / (n - 2))**0.5
    else:
        syx = float('inf')
    
    if st != 0:
        r2 = (st - sr) / st
    else:
        r2 = 0

    return a0, a1, syx, r2

def plot_chapra_interpolation(chapra_func, x_values: np.ndarray, y_values: np.ndarray, num_points: int = 100, especial_point: float = None) -> None:
    # Obter os coeficientes da regressão linear
    n = len(x_values)
    a0, a1, _, _ = chapra_func(n, x_values, y_values)
    
    # Gerar pontos para a linha de regressão
    xplt = np.linspace(x_values.min(), x_values.max(), num_points)
    yplt = a0 + a1 * xplt

    # Plotar os pontos de dados originais
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, 'o', color='#1837E7', label='Pontos', zorder=5)
    
    # Plotar a linha de regressão
    plt.plot(xplt, yplt, '-', color='#37E718', label='Aproximação Linear de chapra')
    
    # Plotar o ponto específico se fornecido
    if especial_point is not None:
        y_especial = a0 + a1 * especial_point
        plt.scatter(especial_point, y_especial, color='#E71837', label=f'Ponto em x = {especial_point}', zorder=5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Aproximação Linear de Chapra')
    plt.legend()
    plt.grid(True)
    plt.show()

x = np.array([0, 20, 40, 60, 80, 100], dtype=float)
y = np.array([26, 48.6, 61.6, 71.2, 74.8, 75.2], dtype=float)
n = len(x)
interpolation_value = 80
"""a0, a1, syx, r2 = chapra(n, x, y)
print(f"a0: {a0}, a1: {a1}, syx: {syx}, r2: {r2}")
print(f"y = {a0 + a1 * interpolation_value} ")
"""
plot_chapra_interpolation(chapra, x, y, 100, interpolation_value)