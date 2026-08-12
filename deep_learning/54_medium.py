"""
Implementing a Simple RNN
 - Medium
 - Deep Learning

Write a Python function that implements a simple Recurrent Neural Network (RNN) cell.
The function should process a sequence of input vectors and produce the final hidden state.
Use the tanh activation function for the hidden state updates.
The function should take as inputs the sequence of input vectors, the initial hidden state,
the weight matrices for input-to-hidden and hidden-to-hidden connections, and the bias vector.
The function should return the final hidden state after processing the entire sequence,
rounded to four decimal places.

Example:
    Input:
        input_sequence = [[1.0], [2.0], [3.0]]
        initial_hidden_state = [0.0]
        Wx = [[0.5]]  # Input to hidden weights
        Wh = [[0.8]]  # Hidden to hidden weights
        b = [0.0]     # Bias
    Output:
        final_hidden_state = [0.9759]
Reasoning:
    The RNN processes each input in the sequence, updating the hidden state at
    each step using the tanh activation function.
"""


import numpy as np

def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float],
) -> list[float]:
    final_hidden_state = np.array(initial_hidden_state)
    for x in input_sequence:
        final_hidden_state = np.tanh(np.dot(Wx, x) + np.dot(Wh, final_hidden_state) + np.array(b))
    return np.round(final_hidden_state, 4)
