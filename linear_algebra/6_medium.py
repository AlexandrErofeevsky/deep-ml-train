"""
Calculate Eigenvalues of a Matrix
 - Medium
 - Linear Algebra

Write a Python function that calculates the eigenvalues of a 2x2 matrix.
The function should return a list containing the eigenvalues, sort values from highest to lowest.

Example:
    Input:
        matrix = [[2, 1], [1, 2]]
    Output:
        [3.0, 1.0]
Reasoning:
    The eigenvalues of the matrix are calculated using the characteristic equation of the matrix,
    which for a 2x2 matrix is
    λ^2 − trace(A) * λ + det(A) = 0, where λ are the eigenvalues.
"""


from math import sqrt


def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    trace = sum([row[i] for i, row in enumerate(matrix)])
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    discriminant = trace ** 2 - 4 * det

    eigenvalues = [
        (trace + multiplier * sqrt(discriminant)) / 2
        for multiplier in [1, -1]
    ]

    # import numpy as np
    # assert set(eigenvalues) == set(np.linalg.eigvals(matrix).tolist())

    return eigenvalues
