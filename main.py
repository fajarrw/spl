import numpy as np

A = np.array([[4, 1, -1],
              [2, 7, 1],
              [1, -3, 12]])
b = np.array([3, 19, 31])

x = np.zeros_like(b, dtype=np.double)
iterations = 25

def jacobi(A, b, x, iterations):
    D = np.diag(np.diag(A))
    R = A - D
    for i in range(iterations):
        x = (b - np.dot(R, x)) / np.diag(A)
    return x

x_jacobi = jacobi(A, b, x, iterations)
print("Jacobi solution:", x_jacobi)

def gauss_elimination(A, b):
    n = len(b)
    for i in range(n):
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i+1:], x[i+1:])) / A[i][i]
    return x

x_gauss_elimination = gauss_elimination(np.copy(A), np.copy(b))
print("Gauss Elimination solution:", x_gauss_elimination)

def gauss_jordan(A, b):
    n = len(b)
    aug_matrix = np.hstack((A, b.reshape(-1, 1)))
    for i in range(n):
        if aug_matrix[i][i] == 0:
            for j in range(i+1, n):
                if aug_matrix[j][i] != 0:
                    aug_matrix[[i, j]] = aug_matrix[[j, i]]  # Swap rows
                    break
        pivot = aug_matrix[i][i]
        aug_matrix[i] = aug_matrix[i] / pivot
        for j in range(n):
            if j != i:
                factor = aug_matrix[j][i]
                aug_matrix[j] = aug_matrix[j] - factor * aug_matrix[i]
    return aug_matrix[:, -1]

x_gauss_jordan = gauss_jordan(np.copy(A), np.copy(b))
print("Gauss-Jordan solution:", x_gauss_jordan)
