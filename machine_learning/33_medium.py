"""
Generate Random Subsets of a Dataset
 - Medium
 - Machine Learning

Write a Python function to generate random subsets of a given dataset for use in ensemble methods like bagging.

The function should take a 2D numpy array X, a 1D numpy array y,
an integer n_subsets specifying how many subsets to generate,
and a boolean replacements controlling sampling behavior.

Sampling Behavior
    If replacements is True: Sample with replacement.
    Each subset should be the same size as the original dataset.
    The same sample can appear multiple times within a subset.

    If replacements is False: Sample without replacement.
    Each subset should be half the size of the original dataset (using integer division).
    Each sample appears at most once within a subset.

Note:
    Replacement refers to whether a sample can be picked more than once within a single subset,
    not across different subsets.

Return Format
    Return a list of n_subsets tuples,
    where each tuple is (X_subset, y_subset) with both elements converted to Python lists.

Example:
    Input:
        np.random.seed(42)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([1, 2, 3, 4, 5])
        n_subsets = 3
        replacements = False
    Output:
        [([[3, 4], [9, 10]], [2, 5]),
         ([[7, 8], [3, 4]], [4, 2]),
         ([[3, 4], [1, 2]], [2, 1])]
Reasoning:
    The function generates three random subsets of the dataset without replacement.
    Since replacements=False, each subset contains half the samples (using integer division),
    and no sample is repeated within a subset.
    Each subset is a tuple of (X_subset, y_subset) converted to lists.
"""


import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    indices = np.arange(len(y))

    if replacements:
        indices = np.random.choice(indices, size=(n_subsets, len(y)))
    else:
        indices = [np.random.choice(
            indices, size=len(y) // 2, replace=False,
        ) for _ in range(n_subsets)]

    return [(X[idx], y[idx]) for idx in indices]
