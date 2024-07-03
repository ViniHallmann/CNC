#Cholesky

import numpy as np

def cholesky(matrix: np.array) -> np.array:
    size = matrix.shape[0]
    lower = np.zeros((size, size))

    for i in range( size ):
        for j in range( i + 1 ):
            sum = 0
            if i == j:
                for k in range( j ):
                    sum += lower[i, k] * lower[j, k]
                lower[i, j] = np.sqrt(matrix[i, i] - sum)
            else:
                for k in range( j ):
                    sum += lower[i, k] * lower[j, k]
                if lower[j, j] > 0:
                    lower[i, j] = ( matrix[i, j] - sum ) / lower[j, j]
    return lower

A = np.array([[4, 2, -4], [2, 10, 4], [-4, 4, 9]], dtype=float)
B = np.array([0, 6, 5], dtype=float)
L = cholesky(A)
y = np.linalg.solve(L, B)
x = np.linalg.solve(L.T, y)
