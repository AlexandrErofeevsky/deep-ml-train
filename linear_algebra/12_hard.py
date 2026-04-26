"""
Singular Value Decomposition (SVD) of 2x2 Matrix
 - Hard
 - Linear Algebra

Write a Python function that computes an approximate Singular Value Decomposition (SVD) of a real 2x2 matrix using one Jacobi rotation.

Input:
    A: a NumPy array of shape (2, 2)

Rules:
    You may use basic NumPy operations (matrix multiplication, transpose, element-wise math, etc.)
    Do NOT call numpy.linalg.svd or any other high-level SVD routine
    Use a single Jacobi rotation step (no iterative refinements)
    Return: A tuple (U, S, Vt) where:

    U is a 2x2 orthogonal matrix (left singular vectors)
    S is a length-2 NumPy array containing the singular values
    Vt is the transpose of the right singular vector matrix V
    The decomposition should approximately satisfy: A = U @ diag(S) @ Vt

Example:
    Input:
        A = np.array([[2, 1], [1, 2]])
    Output:
        U ≈ [[0.707, -0.707], [0.707, 0.707]]
        S = [3.0, 1.0]
        Vt ≈ [[0.707, 0.707], [-0.707, 0.707]]
Reasoning:
    The symmetric matrix [[2,1],[1,2]] has eigenvalues 3 and 1.
    Since it's symmetric, the SVD simplifies: the singular values equal the absolute eigenvalues, and U, V are related to the eigenvectors.
    The decomposition satisfies A = U @ diag(S) @ Vt.
"""

import numpy as np


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.

    Args:
        A: A 2x2 numpy array

    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    b = A.T @ A

    trace_diff = b[0][0] - b[1][1]

    if np.isclose(trace_diff, 0):
        theta = np.pi / 4
    else:
        theta = np.arctan(2 * b[0][1] / trace_diff) / 2

    v = np.array(
        [[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    s = np.sqrt(np.diagonal(v.T @ b @ v))

    u = A @ v @ np.diag(1 / s)

    idx = np.argsort(s)[::-1]
    s = s[idx]
    u = u[:, idx]
    vt = v.T[idx, :]

    return u, s, vt
