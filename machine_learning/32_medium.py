"""
Generate Sorted Polynomial Features
 - Medium
 - Machine Learning

Write a Python function that takes a 2-D NumPy array X and an integer degree,
generates all polynomial feature combinations of the columns of X up to the given degree inclusive,
then sorts the resulting features for each sample from lowest to highest value.
The function should return a new 2-D NumPy array whose rows correspond
to the input samples and whose columns are the ascending-sorted polynomial features.

Example:
    Input:
        X = np.array([[2, 3],
                      [3, 4],
                      [5, 6]])
        degree = 2
    Output:
        [[ 1.  2.  3.  4.  6.  9.]
         [ 1.  3.  4.  9. 12. 16.]
         [ 1.  5.  6. 25. 30. 36.]]
Reasoning:
    For degree = 2, the raw polynomial terms for the first sample are [1, 2, 3, 4, 6, 9].
    Sorting them from smallest to largest yields [1, 2, 3, 4, 6, 9].
    The same procedure is applied to every sample.
"""
import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    list_degrees = list(range(1, degree + 1))
    result = []

    for row in X:
        row_result = [1]
        for d in list_degrees:
            combinations = combinations_with_replacement(row, d)
            row_result.extend([np.prod(c) for c in combinations])
        result.append(sorted(row_result))
    return result
