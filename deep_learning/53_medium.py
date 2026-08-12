"""
Implement Self-Attention Mechanism
 - Medium
 - Deep Learning

Implement the self-attention mechanism, a fundamental component of transformer
models used in NLP and computer vision.

Your task is to implement the self_attention function that computes attention
output given Query (Q), Key (K), and Value (V) matrices.

The self-attention formula is: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

where d_k is the dimensionality of the key vectors (number of columns in K).

    Input:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Output:
        Attention output matrix of shape (seq_len, d_v)

    Steps:
        1. Compute attention scores: scores = Q * K^T / sqrt(d_k)
        2. Apply softmax row-wise to get attention weights (each row should sum to 1)
        3. Compute output: output = attention_weights * V

Note: The helper function compute_qkv is provided to compute Q, K, V from input X and weight matrices.

Example:
    Input:
        Q = np.array([[1, 0], [0, 1]])
        K = np.array([[1, 0], [0, 1]])
        V = np.array([[1, 2], [3, 4]])

        output = self_attention(Q, K, V)
    Output:
        [[1.660477, 2.660477], [2.339523, 3.339523]]
Reasoning:
    1. Compute scores: Q @ K.T / sqrt(2) = [[0.707, 0], [0, 0.707]]
    2. Apply softmax row-wise: [[0.66, 0.34], [0.34, 0.66]]
    3. Multiply by V: attention_weights @ V gives the final contextualized output
"""

import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V


def softmax(x, axis=-1):
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp


def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    return softmax(Q @ K.T / np.sqrt(K.shape[1])) @ V
