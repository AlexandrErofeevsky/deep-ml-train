"""
Matrix Transformation
 - Medium
 - Linear Algebra

Write a Python function that transforms a given matrix A using the operation
T^(−1)AS, where T and S are invertible matrices.
The function should first validate if the matrices T and S are invertible,
and then perform the transformation.
In cases where there is no solution return -1

Example:
    Input:
        A = [[1, 2], [3, 4]], T = [[2, 0], [0, 2]], S = [[1, 1], [0, 1]]
    Output:
        [[0.5,1.5],[1.5,3.5]]
Reasoning:
    The matrices T and S are used to transform matrix A by computing T^(−1)AS.
"""


import numpy as np


def transform_matrix(
    A: list[list[int | float]],
    T: list[list[int | float]],
    S: list[list[int | float]],
) -> list[list[int | float]] | int:
    def is_invertible(matrix):
        if matrix.shape[0] != matrix.shape[1]:
            return False
        return not np.isclose(np.linalg.det(matrix), 0)

    if not (is_invertible(np.array(T)) and is_invertible(np.array(S))):
        return -1

    transformed_matrix = np.dot(np.dot(np.linalg.inv(T), A), S).tolist()

    return transformed_matrix
