"""
Principal Component Analysis (PCA) Implementation
 - Medium
 - Machine Learning

Write a Python function that performs Principal Component Analysis (PCA) from scratch.
The function should take a 2D NumPy array as input,
where each row represents a data sample and each column represents a feature.
The function should standardize the dataset, compute the covariance matrix,
find the eigenvalues and eigenvectors, and return the principal components
(the eigenvectors corresponding to the largest eigenvalues).
The function should also take an integer k as input,
representing the number of principal components to return.

Sign Convention: Eigenvectors can point in either direction (if v is valid, so is -v).
To ensure consistent results across different environments,
apply this rule: for each eigenvector, find its first element with absolute value > 1e-10;
if that element is negative, multiply the entire eigenvector by -1.

Note: Use np.linalg.eigh for eigendecomposition since covariance matrices are symmetric.
This provides more numerically stable and consistent results than np.linalg.eig.

Example:
    Input:
        data = np.array([[1, 2], [3, 4], [5, 6]]),
        k = 1
    Output:
        [[0.7071], [0.7071]]
Reasoning:
    The data lies perfectly along a diagonal line (y = x + 1).
    After standardization, the direction of maximum variance is along [1, 1]
    (normalized to [0.7071, 0.7071]).
    This single principal component captures 100% of the variance.
"""

import numpy as np


def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.

    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return

    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    data_std = (data - data.mean(axis=0)) / data.std(axis=0)

    cov_matr = np.cov(data_std, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matr)

    idxs = np.argsort(eigenvalues)[::-1][:k]

    result_vec = eigenvectors[:, idxs]

    abs_vec = np.abs(result_vec)

    mask = abs_vec > 1e-10
    first_nonzero = np.argmax(mask, axis=0)

    sign_vals = result_vec[first_nonzero, np.arange(result_vec.shape[1])]

    signs = np.where(sign_vals < 0, -1, 1)

    return np.round(result_vec * signs, 4)


print(pca(np.array([[1, 6], [2, 4], [3, 2]]), k=1).tolist())

