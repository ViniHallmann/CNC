#Fatoração LU
import numpy as np

def create_identity_matrix(size: int) -> np.array:
    return np.eye(size, dtype=float)

def partial_pivot( matrix: np.array, pivot_row: int ) -> np.array:
    max_index = np.argmax( abs( matrix[pivot_row:, pivot_row] ) ) + pivot_row
    if pivot_row != max_index:
        matrix[[pivot_row, max_index]] = matrix[[max_index, pivot_row]]
    return matrix

def flu( matrix: np.array, use_pivot: bool = False ) -> list[np.array, np.array]:
    U = np.copy( matrix )
    size = U.shape[0]
    L = create_identity_matrix( size )
    
    for pivot_row in range( size - 1 ):
        if use_pivot:
            upper_matrix = partial_pivot(upper_matrix, pivot_row)
        for target_row in range( pivot_row + 1, size ):
            multiplier = U[target_row, pivot_row] / U[pivot_row, pivot_row]
            L[target_row, pivot_row] = multiplier
            
            for col in range( pivot_row + 1, size ):
                U[target_row, col] -= multiplier * U[pivot_row, col]
                
            U[target_row, pivot_row] = 0
    
    return L, U

def forward_substitution( L: np.array, B ) -> np.array:
  return np.linalg.solve( L, B )

def backward_substitution( U: np.array, Y ) -> np.array:
  return np.linalg.solve( U, Y )

def print_matrix( matrix: np.array, name: str ) -> None:
    print( f"{name} Matrix:" )
    print( np.round( matrix, 3 ) )
    print()

A    = np.array( [[1, 1, 1], [2, 1, -1], [2, -1, 1]] )
B    = np.array( [-2, 1, 3] )
L, U = flu(A)
y    = forward_substitution( L, B )
x    = backward_substitution( U, y )


print_matrix(L, "L")
print_matrix(U, "U")
print_matrix(y, "y")
print_matrix(x, "x")