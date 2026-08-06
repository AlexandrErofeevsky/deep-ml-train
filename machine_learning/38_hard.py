"""
Implement AdaBoost Fit Method
 - Hard
 - Machine Learning

Write a Python function adaboost_fit that implements the fit method for an AdaBoost classifier.
The function should take in a 2D numpy array X of shape (n_samples, n_features)
representing the dataset,a 1D numpy array y of shape (n_samples,) representing
the labels, and an integer n_clf representing the number of classifiers.
The function should initialize sample weights, find the best thresholds for each feature,
calculate the error, update weights, and return a list of classifiers with their parameters.

Example:
    Input:
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([1, 1, -1, -1])
        n_clf = 3

        clfs = adaboost_fit(X, y, n_clf)
        print(clfs)
    Output:
        (example format, actual values may vary):
        [{'polarity': 1, 'threshold': 2, 'feature_index': 0, 'alpha': 0.5},
         {'polarity': -1, 'threshold': 3, 'feature_index': 1, 'alpha': 0.3},
         {'polarity': 1, 'threshold': 4, 'feature_index': 0, 'alpha': 0.2}]
Reasoning:
    The function fits an AdaBoost classifier on the dataset X with the given
    labels y and number of classifiers n_clf. It returns a list of classifiers
    with their parameters, including the polarity, threshold, feature index,
    and alpha values
"""

import numpy as np


def get_clf_pred(clf, X):
    return np.where(
        X[:, clf["feature_index"]] >= clf["threshold"],
        clf["polarity"],
        -clf["polarity"],
    )


def adaboost_error(clf, X, y, w):
    """clf: {'polarity': 1, 'threshold': 2, 'feature_index': 0}"""
    y_pred = get_clf_pred(clf, X)
    return np.sum(w * (y_pred != y))


def adaboost_fit(X, y, n_clf):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))
    clfs = []

    for _ in range(n_clf):
        min_error = np.inf
        min_clf = None
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for thr in thresholds:
                for polarity in [-1, 1]:
                    clf = {'polarity': polarity, 'threshold': thr, 'feature_index': feature}
                    error = adaboost_error(clf, X, y, w)
                    if error < min_error:
                        min_error = error
                        min_clf = clf.copy()

        alpha = np.log((1 - min_error) / (min_error + 1e-10)) / 2
        min_clf["alpha"] = alpha
        clfs.append(min_clf)

        w = w * np.exp(-alpha * y * get_clf_pred(min_clf, X))
        w = w / np.sum(w)

    return clfs


X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([1, 1, -1, -1])
n_clf = 3
clfs = adaboost_fit(X, y, n_clf)
print(clfs)

