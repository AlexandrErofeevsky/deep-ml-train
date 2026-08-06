"""
Implement Gradient Descent Variants with MSE Loss
 - Medium
 - Machine Learning

In this problem, you need to implement a single function that can perform three
variants of gradient descent: Stochastic Gradient Descent (SGD),
Batch Gradient Descent, and Mini-Batch Gradient Descent using Mean Squared Error (MSE) as the loss function.
The function will take an additional parameter to specify which variant to use.

Requirements
Do not shuffle the data; process samples in their original order (index 0, 1, 2, ...)
For Batch GD: use all samples to compute a single gradient update per epoch
For Stochastic GD: iterate through each sample sequentially (i.e., process sample 0, then 1, then 2, etc.) — not randomly selected
For Mini-Batch GD: form batches from consecutive samples without overlap (e.g., for batch_size=2: first batch uses indices [0,1], second batch uses [2,3], etc.)
The n_epochs parameter specifies how many complete passes through the dataset to perform
For each epoch, process all samples according to the specified method
Example:
    Input:
        import numpy as np

        # Sample data
        X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
        y = np.array([2, 3, 4, 5])

        # Parameters
        learning_rate = 0.01
        n_epochs = 1000
        batch_size = 2

        # Initialize weights
        weights = np.zeros(X.shape[1])

        # Test Batch Gradient Descent
        final_weights = gradient_descent(X, y, weights, learning_rate, n_epochs, method='batch')
        # Test Stochastic Gradient Descent
        final_weights = gradient_descent(X, y, weights, learning_rate, n_epochs, method='stochastic')
        # Test Mini-Batch Gradient Descent
        final_weights = gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size, method='mini_batch')
    Output:
        [float, float]
        [float, float]
        [float, float]
Reasoning:
The function should return the final weights after performing the specified variant of gradient descent for the given number of epochs (complete passes through the data).
"""

import numpy as np


def gradient_descent(
    X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch',
):
    """
    Perform gradient descent optimization.

    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')

    Returns:
        Optimized weights
    """
    final_weights = weights.copy()
    match method:
        case "batch":
            for _ in range(n_epochs):
                final_weights -= learning_rate * (X @ final_weights - y) @ X / X.shape[0] * 2
            return final_weights

        case "stochastic":
            for _ in range(n_epochs):
                for x_i, y_i in zip(X, y):
                    final_weights -= learning_rate * 2 * (x_i @ final_weights - y_i) * x_i
            return final_weights

        case "mini_batch":
            for _ in range(n_epochs):
                for b_i in range(0, X.shape[0], batch_size):
                    x_i = X[b_i : b_i + batch_size]
                    y_i = y[b_i : b_i + batch_size]
                    final_weights -= learning_rate * 2 / batch_size * (x_i @ final_weights - y_i) @ x_i
            return final_weights

        case _:
            raise ValueError("Invalid method")


import numpy as np

X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
y = np.array([2, 3, 4, 5])
weights = np.zeros(X.shape[1])
learning_rate = 0.01
n_epochs = 100

# Test Stochastic Gradient Descent
output = gradient_descent(X, y, weights, learning_rate, n_epochs, method='stochastic')
print(output)