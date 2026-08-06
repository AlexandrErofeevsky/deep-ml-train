"""
Implement Reduced Row Echelon Form (RREF) Function
 - Medium
 - Linear Algebra

In this problem, your task is to implement a function that converts a given
matrix into its Reduced Row Echelon Form (RREF). The RREF of a matrix is
a special form where each leading entry in a row is 1, and all other elements
in the column containing the leading 1 are zeros, except for the leading 1 itself.

However, there are some additional details to keep in mind:

Diagonal entries can be 0 if the matrix is reducible
(i.e., the row corresponding to that position can be eliminated entirely).
Some rows may consist entirely of zeros.
If a column contains a pivot (a leading 1), all other entries in that column should be zero.
Your task is to implement the RREF algorithm, which must handle these cases
and convert any given matrix into its RREF.

Example:
    Input:
        import numpy as np

        matrix = np.array([
            [1, 2, -1, -4],
            [2, 3, -1, -11],
            [-2, 0, -3, 22]
        ])

        rref_matrix = rref(matrix)
        print(rref_matrix)
    Output:
        # array([
        #    [ 1.  0.  0. -8.],
        #    [ 0.  1.  0.  1.],
        #    [-0. -0.  1. -2.]
        # ])
Reasoning:
    The given matrix is converted to its Reduced Row Echelon Form (RREF) where
    each leading entry is 1, and all other entries in the leading columns are zero.
"""


import numpy as np


def move_to_pos(matrix, idx_from, idx_to, axis):
    n = matrix.shape[axis]
    order = list(range(n))
    order.pop(idx_from)
    order.insert(idx_to, idx_from)
    if axis == 0:
        return matrix[order]
    else:
        return matrix[:, order]


def rref(matrix):
    matrix = matrix.astype(np.float64)

    for row_idx in range(matrix.shape[0]):
        # TODO: for zero rows
        # for i, row in enumerate(matrix[row_idx:]):
        #     if row.any():
        #         matrix = move_to_pos(matrix, i, row_idx, 0)
        #         break
        # else:
        #     pass

        # ???? task says move columns, but answers force to move rows
        # idx_col = np.argmax(matrix[row_idx] != 0)
        # if idx_col != row_idx:
        #     matrix = move_to_pos(matrix, idx_col, row_idx, 1)

        for idx_col in range(row_idx, matrix.shape[0]):
            if matrix[row_idx:, idx_col].any():
                break
        else:
            break

        rel_idx_col = np.argmax(matrix[row_idx:, idx_col] != 0)
        if rel_idx_col != 0:
            matrix = move_to_pos(matrix, row_idx + rel_idx_col, row_idx, 0)

        matrix[row_idx] /= matrix[row_idx, idx_col]
        for row in matrix[row_idx + 1:]:
            row -= matrix[row_idx] * row[idx_col]

    for row_idx in range(matrix.shape[0] - 1, 0, -1):
        idx_col = np.argmax(matrix[row_idx] != 0)
        for row in matrix[:row_idx]:
            row -= matrix[row_idx] * row[idx_col]

    return matrix


import numpy as np

matrix = np.array([
        [1, 2, -1],
        [2, 4, -1],
        [-2, -4, -3]])

output = rref(matrix)
print(output)
