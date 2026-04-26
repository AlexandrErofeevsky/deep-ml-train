"""
Linear Regression Using Gradient Descent
 - Easy
 - Machine Learning

Write a Python function that performs linear regression using gradient descent.
The function should take NumPy arrays X (features with a column of ones for the intercept) and y (target) as input,
along with the learning rate alpha and the number of iterations.
Return the learned coefficients (weights) as a NumPy array.

Requirements
    - Minimize the Mean Squared Error (MSE) loss function:
        L(θ) = 1 / 2m * ∑_i=1^m (h_θ(x_i)−y_i)**2,
        where h_θ(x)=X_θ is the prediction and m is the number of samples.
      The factor of 1/2 simplifies the gradient calculation.

    - Initialize all weights to zero

    - Use batch gradient descent (use all samples in each iteration)

The input matrix X has shape (m, n)
where m is the number of training examples and n is the number of features (including the bias column of ones).
The target vector y has shape (m,).

Example:
    Input:
        X = np.array([[1, 1], [1, 2], [1, 3]]), y = np.array([3, 5, 7]), alpha = 0.1, iterations = 1000
    Output:
        [1.0, 2.0]
Reasoning:
    The data follows y = 1 + 2x perfectly.
    Starting from theta = [0, 0], gradient descent iteratively updates the weights until converging to approximately [1, 2],
    representing an intercept of 1 and slope of 2.
"""

import numpy as np


def linear_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    iterations: int,
) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations

    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)
    theta = np.zeros((n, 1))

    for _ in range(iterations):
        errors = X @ theta - y
        grad = X.T @ errors / m

        theta = theta - alpha * grad

    return theta.flatten()
