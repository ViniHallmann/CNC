#Newton

import numpy as np

def newton(x0, e, func, jf, iter) -> np.array:
    while max(abs(func(x0))) > e:
        s = np.linalg.solve(jf(x0), -func(x0))
        x1 = x0 + s
        if max(abs(x1 - x0)) < e: return x1
        iter -= 1
        x0 = x1
        if iter == 0: return x1

    return x1

#f = lambda x: np.array([x[0] + x[1] - 3, x[0]**2 + x[1]**2 - 9])
#j = lambda x: np.array([[1, 1], [2*x[0], 2*x[1]]])
#f1 = lambda x: np.array([ 3 * np.sin(x[0]) - 4 * x[1] - 12 * x[2] - 1, 4 * x[0] - 8 * x[1] - 10 * x[2] + 5, 2 * np.exp(x[0]) + 2 * x[1] + 3 * x[2] - 8])
#j1 = lambda x: np.array([ [3 * np.cos(x[0]), -4, -12], [4, -8, -10], [2 * np.exp(x[0]), 2, 3]])
#print(newton([0,0,0], 10**-5, f1, j1, 100))

f = lambda x: np.array([ 3 * np.sin(x[0]) - 4 * x[1] - 12 * x[2] - 1, 4 * x[0] - 8 * x[1] - 10 * x[2] + 5, 2 * np.exp(x[0]) + 2 * x[1] + 3 * x[2] - 8])
jf = lambda x: np.array([ [3 * np.cos(x[0]), -4, -12], [4, -8, -10], [2 * np.exp(x[0]), 2, 3]])

n = newton([0,0,0], 10**-5, f, jf, 100)
print(n)