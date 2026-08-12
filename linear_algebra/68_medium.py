"""
Find the column space of a matrix
 - Medium
 - Linear Algebra

Task: Compute the Column Space of a Matrix
In this task, you are required to implement a function matrix_image(A)
that calculates the column space of a given matrix A.
The column space, also known as the image or span, consists of all linear
combinations of the columns of A. To find this, you'll use concepts from linear
algebra, focusing on identifying independent columns that span the matrix's image.
Your task: Implement the function matrix_image(A) to return the basis vectors
that span the column space of A. These vectors should be extracted from
the original matrix and correspond to the independent columns.

Example:
    Input:
        matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        print(matrix_image(matrix))
    Output:
        [[1, 2],
         [4, 5],
         [7, 8]]
Reasoning:
    The matrix has rank 2, meaning only 2 columns are linearly independent.
    The column space is spanned by the first two column vectors [1, 4, 7] and [2, 5, 8].
    The output matrix contains these two independent columns.
"""


import numpy as np


def matrix_image(A):
    pivot_cols = []

    matrix = np.array(A, dtype=float)
    row = 0
    m, n = A.shape

    for col in range(n):
        pivot = np.argmax(np.abs(matrix[row:, col])) + row if row < m else None
        if pivot is None or abs(matrix[pivot, col]) < 1e-10:
            continue

        if pivot != row:
            matrix[[row, pivot]] = matrix[[pivot, row]]

        matrix[row] = matrix[row] / matrix[row, col]
        for r in range(m):
            if r != row:
                matrix[r] -= matrix[r, col] * matrix[row]

        pivot_cols.append(col)
        row += 1

        if row == m:
            break

    return A[:, pivot_cols]
