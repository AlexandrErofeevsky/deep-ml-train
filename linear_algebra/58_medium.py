"""
Gaussian Elimination for Solving Linear Systems
 - Medium
- Linear Algebra

Task: Implement the Gaussian Elimination Method
Your task is to implement the Gaussian Elimination method, which transforms a
system of linear equations into an upper triangular matrix. This method can then
be used to solve for the variables using backward substitution.

Write a function gaussian_elimination(A, b) that performs Gaussian Elimination
with partial pivoting to solve the system (Ax = b).

The function should return the solution vector (x).

Example:
    Input:
        A = np.array([[2,8,4], [2,5,1], [4,10,-1]], dtype=float)
        b = np.array([2,5,1], dtype=float)

        print(gaussian_elimination(A, b))
    Output:
        [11.0, -4.0, 3.0]
Reasoning:
    The Gaussian Elimination method transforms the system of equations into an
    upper triangular matrix and then uses backward substitution to solve for the variables.
"""

import numpy as np


def move_to_pos(matrix, b, idx_from, idx_to, axis):
    n = matrix.shape[axis]
    order = list(range(n))
    order.pop(idx_from)
    order.insert(idx_to, idx_from)
    if axis == 0:
        return matrix[order], b[order]
    else:
        return matrix[:, order], b[order]


def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.

    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    x = np.zeros_like(b)
    matrix = A.astype(np.float64)
    for row_idx in range(matrix.shape[0]):
        idx_col = row_idx
        rel_idx_col = np.argmax(matrix[row_idx:, idx_col])
        if rel_idx_col != 0:
            matrix, b = move_to_pos(matrix, b, row_idx + rel_idx_col, row_idx, 0)

        for row_idx_curr in range(row_idx + 1, len(matrix)):
            mult = matrix[row_idx_curr][idx_col] / matrix[row_idx, idx_col]
            matrix[row_idx_curr] -= matrix[row_idx] * mult
            b[row_idx_curr] -= b[row_idx] * mult

    x[-1] = b[-1] / matrix[-1, -1]
    for row_idx in range(matrix.shape[0] - 2, -1, -1):
        x[row_idx] = (b[row_idx] - x @ matrix[row_idx]) / matrix[row_idx, row_idx]
    return x
