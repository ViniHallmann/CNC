import numpy as np
import matplotlib.pyplot as plt

def Spline(n: int, x: np.array, y: np.array, interpolation_value: float) -> float:
    info: int
    if n < 3:
        info = -1
        return info

    ordened: bool = True
    for i in range(n - 1):
        ordened = ordened and x[i] <= x[i + 1]

    if not ordened:
        info = -2
        return info

    info = 0
    m = n - 1
    h1 = x[1] - x[0]
    delta1 = (y[1] - y[0]) / h1
    e = np.zeros(m - 1)
    d = np.zeros(m - 1)
    s2 = np.zeros(n)
    for i in range(m - 1):
        ip1 = i + 1
        ip2 = i + 2
        h2 = x[ip2] - x[ip1]
        delta2 = (y[ip2] - y[ip1]) / h2
        e[i] = h2
        d[i] = 2 * (h1 + h2)
        s2[ip1] = 6 * (delta2 - delta1)
        h1 = h2

    for i in range(1, m - 1):
        t = e[i - 1] / d[i - 1]
        d[i] = d[i] - t * e[i - 1]
        s2[i + 1] = s2[i + 1] - t * s2[i]

    s2[m - 1] = s2[m - 1] / d[m - 2]
    for i in range(m - 3, -1, -1):
        s2[i + 1] = (s2[i + 1] - e[i] * s2[i + 2]) / d[i]

    s2[0] = 0
    s2[m] = 0

    for i in range(m):
        if x[i] <= interpolation_value <= x[i + 1]:
            h = x[i + 1] - x[i]
            A = (x[i + 1] - interpolation_value) / h
            B = (interpolation_value - x[i]) / h
            result = A * y[i] + B * y[i + 1] + ((A ** 3 - A) * s2[i] + (B ** 3 - B) * s2[i + 1]) * (h ** 2) / 6
            return result

    return info

x = np.array([0, 20, 40, 60, 80, 100], dtype=float)
y = np.array([26, 48.6, 61.6, 71.2, 74.8, 75.2], dtype=float)
n = len(x)
interpolation_value = 40

result = Spline(n, x, y, interpolation_value)
print(f"Spline value at {interpolation_value}: {result}")

"""
ln_vol = np.log(vol)
a0_abx, a1_abx, _, _ = chapra(len(hour), hour, ln_vol)
a_abx = np.exp(a0_abx)
b_abx = np.exp(a1_abx)

ln_hour = np.log(hour[1:])  
ln_vol_excl_0 = ln_vol[1:]
a0_axb, b_axb, _, _ = chapra(len(ln_hour), ln_hour, ln_vol_excl_0)
a_axb = np.exp(a0_axb)

hour_fit = np.linspace(0, 6, 100)
vol_abx_fit = a_abx * (b_abx ** hour_fit)
vol_axb_fit = a_axb * (hour_fit ** b_axb)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

axs[0].plot(hour, vol, 'o', label='Dados Originais')
axs[0].plot(hour_fit, vol_abx_fit, '-', label=f'Ajuste: $y = {a_abx:.2f} \\cdot ({b_abx:.2f})^x$')
axs[0].set_xlabel('Horas')
axs[0].set_ylabel('Nº de bactérias por vol. unitário')
axs[0].set_title('Ajuste Exponencial')
axs[0].legend()
axs[0].grid(True)

# Plotagem para y = ax^b
axs[1].plot(hour, vol, 'o', label='Dados Originais')
axs[1].plot(hour_fit, vol_axb_fit, '-', label=f'Ajuste: $y = {a_axb:.2f} \\cdot x^{{{b_axb:.2f}}}$')
axs[1].set_xlabel('Horas')
axs[1].set_ylabel('Nº de bactérias por vol. unitário')
axs[1].set_title('Ajuste Potencial')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()



hours_to_predict = 7

# Para y = ab^x
predicted_abx = a_abx * (b_abx ** hours_to_predict)

# Para y = ax^b
predicted_axb = a_axb * (hours_to_predict ** b_axb)

# Imprimindo os resultados
print(f"Previsão para y = ab^x com x = 7 horas: {predicted_abx}")
print(f"Previsão para y = ax^b com x = 7 horas: {predicted_axb}")"""