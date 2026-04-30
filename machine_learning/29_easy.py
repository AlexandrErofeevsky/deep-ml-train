"""
Random Shuffle of Dataset
 - Easy
 - Machine Learning

Write a Python function to perform a random shuffle of the samples in two numpy arrays,
X and y, while maintaining the corresponding order between them.
The function should have an optional seed parameter for reproducibility.

Example:
    Input:
        X = np.array([[1, 2],
                      [3, 4],
                      [5, 6],
                      [7, 8]])
        y = np.array([1, 2, 3, 4])
    Output:
        (array([[5, 6],
                [1, 2],
                [7, 8],
                [3, 4]]),
        array([3, 1, 4, 2]))
Reasoning:
    The samples in X and y are shuffled randomly,
    maintaining the correspondence between the samples in both arrays.
"""


import numpy as np


def shuffle_data(X, y, seed=None):
    np.random.seed(seed)
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    return X[indices], y[indices]
