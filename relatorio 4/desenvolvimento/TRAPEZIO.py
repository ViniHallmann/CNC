#Algoritmo Trapézio - Chapra

trap = lambda h, f0, f1: h*(f0 + f1)/2

def multiTrap(h, n, f):
    sum = f[0]
    for i in range(1, n - 1):
        sum += 2 * f[i]
    sum += f[n]
    return h * sum / 2