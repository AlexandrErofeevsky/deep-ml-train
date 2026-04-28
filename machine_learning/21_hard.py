"""
Pegasos Kernel SVM Implementation
 - Hard
 - Machine Learning

Write a Python function that implements a deterministic version of the Pegasos algorithm
to train a kernel SVM classifier from scratch.
The function should take a dataset
(as a 2D NumPy array where each row represents a data sample and each column represents a feature),
a label vector (1D NumPy array where each entry corresponds to the label of the sample),
and training parameters such as the choice of kernel (linear or RBF),
regularization parameter (lambda), and the number of iterations.
Note that while the original Pegasos algorithm is stochastic
(it selects a single random sample at each step),
this problem requires using all samples in every iteration (i.e., no random sampling).
The function should perform binary classification and return the model's alpha
coefficients as a list and bias as a float.

Example:
    Input:
        data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]]),
        labels = np.array([1, 1, -1, -1]),
        kernel = 'linear',
        lambda_val = 0.01,
        iterations = 100
    Output:
        ([2.0, 2.0, 6.0, 1.0], 36.2027)
Reasoning:
    Using the linear kernel,
    the Pegasos algorithm iteratively updates the alpha coefficients and bias.
    Points that violate the margin constraint (y * f(x) < 1) get their alphas updated.
    After 100 iterations, the first and last two points become support vectors with large alpha values,
    while the second point (which is well-classified) has alpha = 0.
"""

import numpy as np


def pegasos_kernel_svm(
    data: np.ndarray,
    labels: np.ndarray,
    kernel='linear',
    lambda_val=0.01,
    iterations=100,
    sigma=1.0,
) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.

    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')

    Returns:
        Tuple of (alphas, bias) where alphas is a list and bias is a float
    """
    if kernel == "linear":
        kernel_func = lambda x, y: x @ y
    else:
        kernel_func = lambda x, y: np.exp(- np.linalg.norm(x - y, axis=1) ** 2 / (2 * sigma**2))

    alphas = np.zeros(len(labels))
    bias = 0
    for t in range(1, iterations + 1):
        l_r = 1 / (t * lambda_val)

        for i, (x_i, y_i) in enumerate(zip(data, labels)):
            f_i = sum(alphas * labels * kernel_func(data, x_i)) + bias

            if y_i * f_i < 1:
                bias = bias + l_r * y_i
                alphas[i] = (1 - l_r * lambda_val) * alphas[i] + l_r
            else:
                alphas[i] = (1 - l_r * lambda_val) * alphas[i]
    return alphas, bias
