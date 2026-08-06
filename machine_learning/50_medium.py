"""
Implement Lasso Regression using ISTA
Medium
Machine Learning

Implement Lasso Regression using Proximal Gradient Descent (ISTA). Lasso Regression uses L1 regularization, which adds a penalty equal to the absolute value of the coefficients to the loss function. This encourages sparsity by shrinking some coefficients to exactly zero, performing automatic feature selection.

The objective function is:

J(w, b) = 1/2n * ∑[i=1, n](y_i − y^i)**2 + α ∑[j=1, p]∣w_j∣


Since the L1 penalty is non-differentiable at zero, we use the Iterative
Shrinkage-Thresholding Algorithm (ISTA), which alternates between:

 1. A gradient descent step on the smooth MSE loss
 2. A soft-thresholding step that handles the L1 penalty


The soft-thresholding operator is defined as:

S(w, λ) = sign(w) ⋅ max(∣w∣ − λ, 0)

This operator shrinks weights toward zero and sets small weights exactly to zero when |w| ≤ λ.

Example:
    Input:
        X = np.array([[1, 0.01], [2, 0.02], [3, 0.03], [4, 0.04], [5, 0.05]])
        y = np.array([2, 4, 6, 8, 10])
        weights, bias = l1_regularization_gradient_descent(X, y, alpha=0.5, learning_rate=0.01, max_iter=1000)
    Output:
        weights = array([1.96, 0.0]), bias ≈ 0.1
Reasoning:
    The data follows y ≈ 2*X[:,0]. The second feature (0.01, 0.02, ...) is essentially noise.
    ISTA's soft-thresholding sets the second weight to exactly zero, demonstrating Lasso's
    feature selection capability. The first weight is close to 2 (slightly shrunk by regularization).
"""

import numpy as np


def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.

    S(w, λ) = sign(w) * max(|w| - λ, 0)

    Args:
        w: Input array
        threshold: Threshold value λ

    Returns:
        Soft-thresholded array where:
        - Values with |w| > λ are shrunk toward zero by λ
        - Values with |w| ≤ λ become exactly zero
    """
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)


def l1_regularization_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).

    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change

    Returns:
        tuple: (weights, bias)

    Note: The bias term is NOT regularized.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):
        y_pred = X @ weights + bias
        w_temp = weights - learning_rate / n_samples * X.T @ (y_pred - y)
        new_weights = soft_threshold(w_temp, learning_rate * alpha)
        new_bias = bias - learning_rate / n_samples * np.sum(y_pred - y)
        if np.linalg.norm(new_weights - weights) < tol:
            weights = new_weights
            bias = new_bias
            break

        weights = new_weights
        bias = new_bias
    return weights, bias
