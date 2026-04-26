"""
Implement K-Fold Cross-Validation
 - Medium
 - Machine Learning

Implement a function to generate train and test index splits for K-Fold Cross-Validation.
Your task is to divide dataset indices into k folds and return a list of train-test index pairs for each fold.

Input:
    n_samples: The total number of samples in the dataset (integer)
    k: The number of folds (default 5)
    shuffle: Whether to shuffle indices before splitting (default True)

Output:
    A list of k tuples, where each tuple contains (train_indices, test_indices) as Python lists of integers
Requirements:
    Split the indices into k roughly equal folds. If n_samples is not divisible by k,
    distribute the extra samples to the first folds (e.g., with 10 samples and k=3, fold sizes would be [4, 3, 3]).
    For each fold iteration, use that fold as the test set and combine all other folds as the training set.
    When shuffle=True, use np.random.shuffle() to randomize indices before splitting
     (the random seed will be set externally before calling your function).
    When shuffle=False, keep indices in their original order [0, 1, 2, ..., n_samples-1].

Example:
    Input:
        k_fold_cross_validation(n_samples=10, k=5, shuffle=False)
    Output:
        [([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]), ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]), ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]), ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]), ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])]
Reasoning:
    With 10 samples and k=5, we create 5 folds of 2 samples each.
    In each iteration, one fold becomes the test set (2 samples)
    and the remaining 4 folds become the training set (8 samples).
    Every sample appears in the test set exactly once across all iterations.
"""

import numpy as np


def k_fold_cross_validation(
    n_samples: int, k: int = 5, shuffle: bool = True,
) -> list[tuple[list[int], list[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.

    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)

    Returns:
        List of (train_indices, test_indices) tuples
    """
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    folds = np.array_split(indices, k)

    result = []

    for i, fold in enumerate(folds):
        result.append(
            (
                np.concatenate(folds[:i] + folds[i+1:]).tolist(),
                fold.tolist(),
            )
        )

    return result
