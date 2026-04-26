"""
Rotary Positional Embeddings (RoPE)
 - Medium
 - Deep Learning

Implement Rotary Positional Embeddings (RoPE),
a technique used in modern transformer architectures to encode position information directly into query and key vectors.

Write a function apply_rope(x, positions, base) that applies rotary positional embeddings to an input tensor.

Parameters
    x: A numpy array of shape (seq_len, d) representing token embeddings, where d is even.
    positions: A numpy array of shape (seq_len,) containing the integer position index for each token.
    base: A float (default 10000.0) used as the base for computing rotation frequencies.
Returns
    A numpy array of the same shape (seq_len, d) with rotary positional embeddings applied.

Key Concept
    RoPE works by pairing consecutive dimensions of the embedding and applying a 2D rotation to each pair.
    The rotation angle for each pair depends on the token's position and a frequency that varies across dimension pairs.
    Low-frequency rotations encode coarse positional information while high-frequency rotations encode fine-grained information.

The frequency for each dimension pair is derived from the base parameter,
and each pair at index i (where i = 0, 1, ..., d/2 - 1) uses a different frequency.
The rotation angle at a given position is the product of the position index and the corresponding frequency.

Example:
    Input:
        x = np.array([[1.0, 0.0], [0.0, 1.0]]), positions = np.array([0, 1]), base = 10000.0
    Output:
        [[1.0, 0.0], [-0.8415, 0.5403]]
Reasoning:
    With d=2, there is one dimension pair with frequency theta_0 = 1/10000^(0/2) = 1.0.
    At position 0, the rotation angle is 0, so the vector [1.0, 0.0] is unchanged.
    At position 1, the angle is 1.0 radian.
    The pair (0.0, 1.0) is rotated: new_even = 0cos(1) - 1sin(1) = -0.8415, new_odd = 0sin(1) + 1cos(1) = 0.5403.
"""

import numpy as np


def apply_rope(
    x: np.ndarray, positions: np.ndarray, base: float = 10000.0,
) -> np.ndarray:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.

    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)

    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    seq_len, d = x.shape
    n_pairs = d // 2

    i = np.arange(n_pairs, dtype=np.float32)
    freqs = base ** (-2 * i / d)

    theta = np.outer(positions, freqs)
    cos = np.cos(theta)
    sin = np.sin(theta)

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    result_even = x_even * cos - x_odd * sin
    result_odd = x_even * sin + x_odd * cos

    result = np.empty_like(x)
    result[:, 0::2] = result_even
    result[:, 1::2] = result_odd

    return result


x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
positions = np.array([0, 1])
result = apply_rope(x, positions)
print(np.round(result, 4).tolist())
