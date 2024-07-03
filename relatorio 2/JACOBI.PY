#Jacobi

import numpy as np

def gauss_jacobi( matrix: np.array, b_vector: np.array, initial_guess: np.array, tolerance: float, max_iterations: int ) -> np.array:
    current_solution = np.copy( initial_guess )
    size = matrix.shape[0]

    for _ in range( max_iterations ):
        new_solution = np.copy( current_solution )
        
        for i in range( size ):
            sum_values = b_vector[i]
            for j in range( size ):
                if j != i:
                    sum_values -= matrix[i, j] * current_solution[j]
            
            new_solution[i] = sum_values / matrix[i, i]
        
        if np.linalg.norm( new_solution - current_solution, np.inf ) < tolerance:
            return new_solution
        
        current_solution = np.copy( new_solution )


