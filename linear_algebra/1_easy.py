"""
Matrix-Vector Dot Product
 - Easy
 - Linear Algebra

Write a Python function that computes the dot product of a matrix and a vector.
The function should return a list representing the resulting vector if the operation is valid, or -1 if the matrix and vector dimensions are incompatible.
A matrix (a list of lists) can be dotted with a vector (a list) only if the number of columns in the matrix equals the length of the vector.
For example, an n x m matrix requires a vector of length m.

You may assume the input matrix is a valid (non-jagged) list of lists and the vector is a non-empty list.

Example:
    Input:
        a = [[1, 2], [2, 4]], b = [1, 2]
    Output:
        [5, 10]
Reasoning:
    Row 1: (1 * 1) + (2 * 2) = 1 + 4 = 5
    Row 2: (2 * 1) + (4 * 2) = 2 + 8 = 10
"""


def matrix_dot_vector(
    a: list[list[int | float]],
    b: list[int | float],
) -> list[int | float] | int:
    """
    Return a list where each element is the dot product of a row of 'a' with 'b'.
    If the number of columns in 'a' does not match the length of 'b', return -1.
    """
    if len(a[0]) != len(b):
        return -1

    return [
        sum([el_mat * el_vec for el_mat, el_vec in zip(col, b)]) for col in a
    ]
