#Método de Euler - Filho

import numpy as np



def euler( func, interval: np.array, y0: int, m: int ) -> np.array:
    h = ( interval[1] - interval[0] ) / m
    x = interval[0]
    y = y0
    vet_x = [x]
    vet_y = [y]
    
    for i in range( 1, m ):
        x = interval[0] + i * h
        y = y + h * func( x, y )
        vet_x.append( x )
        vet_y.append( y )

    return vet_x, vet_y

def main():
    f = lambda x, y: -x * y
    interval = [0, 1]
    y0 = 1
    m = 10
    x, y = euler( f, interval, y0, m)
    for i in range(len(x)):
        print(f"x = {round(x[i], 3):<10} y = {round(y[i], 3):<10}")

if __name__ == "__main__":
    main()