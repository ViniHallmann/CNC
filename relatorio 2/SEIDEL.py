#Gauss-Seidel 
import numpy as np

def gauss_seidel(matrix: np.array, b_vector: np.array, initial_guess: np.array, tolerance: float, max_iterations: int) -> np.array:
    current_solution = np.copy(initial_guess)
    size = matrix.shape[0]
    iteration_count = 0

    while iteration_count < max_iterations:
        iteration_count += 1
        
        for row in range(size):
            sum_values = b_vector[row]
            for col in range(size):
                if col != row:
                    sum_values -= matrix[row, col] * current_solution[col]
            current_solution[row] = sum_values / matrix[row, row]
            print(current_solution[row], matrix[row, row])
        
        if np.linalg.norm(current_solution - initial_guess, np.inf) < tolerance:
            return current_solution
        
        initial_guess = np.copy(current_solution)

    raise ValueError('Número máximo de iterações excedido.')

def print_solution(solution: np.array) -> None:
    for index, value in enumerate(solution):
        print(f"x{index+1} = {np.round(value, 3)}")


A = np.array([[5,1,1],[-1,3,-1],[1,2,10]], dtype=float)
b = np.array([50, 10, -30], dtype=float)
x0 = np.zeros(len(b), dtype=float)
tol = 10**-3
max_iter = 100

solution = gauss_seidel(A, b, [0,0,0], tol, max_iter)
print_solution(solution)
