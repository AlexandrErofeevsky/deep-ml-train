"""
Gauss-Seidel Method for Solving Linear Systems
 - Medium
 - Linear Algebra

Task: Implement the Gauss-Seidel Method
Your task is to implement the Gauss-Seidel method, an iterative technique for
solving a system of linear equations (Ax = b).

The function should iteratively update the solution vector (x) by using the most
recent values available during the iteration process.

Write a function gauss_seidel(A, b, n, x_ini=None) where:
 - A is a square matrix of coefficients,
 - b is the right-hand side vector,
 - n is the number of iterations,
 - x_ini is an optional initial guess for (x) (if not provided, initialize with zeros).

The function should return the approximated solution vector (x) after performing the specified number of iterations.

Assumptions:
 - The matrix A is diagonally dominant (ensures convergence)
 - All diagonal elements of A are non-zero
 - The system has a unique solution

Example:
    Input:
        A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]], dtype=float)
        b = np.array([4, 7, 3], dtype=float)

        n = 100
        print(gauss_seidel(A, b, n))
    Output:
        # [0.2, 1.4, 0.8]  (Approximate, values may vary depending on iterations)
Reasoning:
    The Gauss-Seidel method iteratively updates the solution vector (x) until convergence.
    The output is an approximate solution to the linear system.
"""


import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    if x_ini is None:
        x_ini = np.zeros_like(b)
    x = x_ini
    for _ in range(n):
        for i in range(len(x)):
            x[i] = (b[i] - A[i, : i] @ x[: i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def gauss_seidel_matrix(A, b, n, x_ini=None):
    if x_ini is None:
        x_ini = np.zeros_like(b)
    x = x_ini
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    inv_matr = np.linalg.inv(D + L)
    for _ in range(n):
        x = inv_matr @ (b - U @ x)
    return x


# TODO: Combine matrix + raw implementation
