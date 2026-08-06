"""
Convert Vector to Diagonal Matrix
 - Easy
 - Linear Algebra

Write a Python function to convert a 1D numpy array into a diagonal matrix.
The function should take in a 1D numpy array x and return a 2D numpy array
representing the diagonal matrix.

Example:
    Input:
        x = np.array([1, 2, 3])
        output = make_diagonal(x)
        print(output)
    Output:
        [[1. 0. 0.]
        [0. 2. 0.]
        [0. 0. 3.]]
Reasoning:
    The input vector [1, 2, 3] is converted into a diagonal matrix where
    the elements of the vector form the diagonal of the matrix.
"""


import numpy as np


def make_diagonal(x):
	return np.diag(x)
