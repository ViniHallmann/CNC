#Eliminação de GAUSS
import numpy as np

A = np.array([[1.5, 5.4, 3.3],[4.2,2.3,4.5],[2.7,5.7,7.8]])
B = np.array([[10], [11.7], [8.9]])
    
def create_concatenated_matrix( A:np.array, B:np.array ) -> np.array: return np.concatenate( ( A,B ), axis=1, dtype=float )

def partial_pivoting( matrix ) -> np.array:
    num_equations = len( matrix )
    for current_row in range( num_equations ):
        max_index = np.argmax(abs(matrix[current_row:num_equations, current_row])) + current_row
        if current_row != max_index: matrix[[current_row, max_index]] = matrix[[max_index, current_row]]
    return matrix

def forward_elimination( matrix:np.array ) -> np.array:
    num_equations = matrix.shape[0]
    for current_row in range( num_equations ):
        for target_row in range( current_row+1, num_equations ):
            multiplier = matrix[target_row][current_row]/matrix[current_row][current_row]
            matrix[target_row] = matrix[target_row] - ( multiplier * matrix[current_row] )
    return matrix

def backward_substitution( matrix ) -> np.array:
    num_equations = matrix.shape[0]
    solutions = np.zeros( num_equations )
    solutions[-1] = matrix[-1, -1] / matrix[-1, -2]
    for row in range( num_equations - 2, -1, -1 ):
        solutions[row] = matrix[row, -1]
        for col in range( row + 1, num_equations ):
            solutions[row] -= matrix[row, col] * solutions[col]
        solutions[row] /= matrix[row, row]
    return solutions

def print_solutions( solutions ) -> None:
    for i, solution in enumerate(solutions):
        print(f"x{i+1} = {np.round(solution, 3)}")

def gauss_elimination( A:np.array, B:np.array, use_pivot:bool = False ) -> np.array:
    concatenated_matrix = create_concatenated_matrix( A, B )
    if use_pivot: concatenated_matrix = partial_pivoting( concatenated_matrix )
    concatenated_matrix = forward_elimination( concatenated_matrix )
    solutions = backward_substitution( concatenated_matrix )
    return solutions

print_solutions(gauss_elimination(A,B, True))

