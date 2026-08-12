"""
Implement Compressed Row Sparse Matrix (CSR) Format Conversion
 - Easy
 - Linear Algebra

Task: Convert a Dense Matrix to Compressed Row Sparse (CSR) Format
Your task is to implement a function that converts a given dense matrix into the
Compressed Row Sparse (CSR) format, an efficient storage representation for
sparse matrices. The CSR format only stores non-zero elements and their positions,
significantly reducing memory usage for matrices with a large number of zeros.

Write a function compressed_row_sparse_matrix(dense_matrix) that takes a 2D list
dense_matrix as input and returns a tuple containing three lists:
 - Values array: List of all non-zero elements in row-major order.
 - Column indices array: Column index for each non-zero element in the values array.
 - Row pointer array: Cumulative number of non-zero elements per row, indicating the start of each row in the values array.

Example:
    Input:
        dense_matrix = [
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [3, 0, 4, 0],
            [1, 0, 0, 5]
        ]

        vals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)
        print("Values array:", vals)
        print("Column indices array:", col_idx)
        print("Row pointer array:", row_ptr)
    Output:
        Values array: [1, 2, 3, 4, 1, 5]
        Column indices array: [0, 1, 0, 2, 0, 3]
        Row pointer array: [0, 1, 2, 4, 6]
Reasoning:
    The dense matrix is converted to CSR format with the values array containing
    non-zero elements, column indices array storing the corresponding column index,
    and row pointer array indicating the start of each row in the values array.
"""


import numpy as np


def compressed_row_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

    :param dense_matrix: 2D list representing a dense matrix
    :return: A tuple containing (values array, column indices array, row pointer array)
    """
    row_pointer = 0
    vals, col_idx, row_ptr = [], [], [0]

    for i, row in enumerate(dense_matrix):
        for j, el in enumerate(row):
            if el:
                vals.append(el)
                col_idx.append(j)
                row_pointer += 1
        row_ptr.append(row_pointer)

    return vals, col_idx, row_ptr
