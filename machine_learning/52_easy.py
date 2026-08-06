"""
Implement Recall Metric in Binary Classification
 - Easy
 - Machine Learning

Task: Implement Recall in Binary Classification
Your task is to implement the recall metric in a binary classification setting.
Recall is a performance measure that evaluates how effectively a machine learning
model identifies positive instances from all the actual positive cases in a dataset.

You need to write a function recall(y_true, y_pred) that calculates the recall metric.
The function should accept two inputs:
 - y_true: A list of true binary labels (0 or 1) for the dataset.
 - y_pred: A list of predicted binary labels (0 or 1) from the model.

Your function should return the recall value as a float.
If the denominator (TP + FN) is zero, return 0.0 to avoid division by zero.

Example:
    Input:
        import numpy as np

        y_true = np.array([1, 0, 1, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 0, 1])

        print(recall(y_true, y_pred))
    Output:
        0.75
Reasoning:
    There are 4 actual positive instances in y_true.
    The model correctly identified 3 of them, giving a recall of 0.75.
"""

import numpy as np


def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.

    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)

    Returns:
        Recall value as a float
    """
    return np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true)
