"""
SVD of a 2x2 Matrix
 - Hard
 - Linear Algebra

Given a 2x2 matrix, write a Python function to compute its Singular Value Decomposition (SVD).
The function should return matrices U, s, and V such that A = U @ diag(s) @ V.

Do not use numpy.linalg.svd or other built-in SVD functions.

Returns:
    U: 2x2 orthogonal matrix (left singular vectors)
    s: 1D array of 2 singular values (non-negative)
    V: 2x2 matrix (right singular vectors)
Example:
    Input:
        A = np.array([[-10, 8], [10, -1]])
    Output:
        U, s, V such that U @ diag(s) @ V ≈ A
Reasoning:
    The SVD decomposes A into orthogonal matrices U and V, and singular values s.
    The reconstruction U @ diag(s) @ V equals the original matrix A.
"""


import numpy as np


def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.

    Args:
        A: 2x2 numpy array

    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    x = [A[0][0] + A[1][1], A[0][0] - A[1][1]]
    y = [A[1][0] + A[0][1], A[1][0] - A[0][1]]

    h_1 = np.sqrt(y[0] ** 2 + x[0] ** 2) + 10e-17
    h_2 = np.sqrt(y[1] ** 2 + x[1] ** 2) + 10e-17

    singular_values = np.array([(h_1 + h_2) / 2, np.abs(h_1 - h_2) / 2])

    t = np.array([x[0] / h_1, x[1] / h_2])

    angle = np.atan2(t[0], t[1])

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    u = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])

    v = np.diag(1 / singular_values) @ u.T @ A

    return u ,singular_values, v


# Symmetric matrix
U, s, V = svd_2x2(np.array([[2, 1], [1, 2]]))
result = U @ np.diag(s) @ V
print(np.allclose(result, [[2, 1], [1, 2]]))
