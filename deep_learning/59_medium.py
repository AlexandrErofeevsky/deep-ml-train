"""
Implement Long Short-Term Memory (LSTM) Network
 - Medium
 - Deep Learning

Task: Implement Long Short-Term Memory (LSTM) Network
Your task is to implement an LSTM network that processes a sequence of inputs
and produces the final hidden state and cell state after processing all inputs.

Write a class LSTM with the following methods:
     - __init__(self, input_size, hidden_size): Initializes the LSTM with random
     weights and zero biases.
     - forward(self, x, initial_hidden_state, initial_cell_state): Processes a
     sequence of inputs and returns the hidden states at each time step, as well
     as the final hidden state and cell state.

The LSTM should compute the forget gate, input gate, candidate cell state, and output gate at each time step to update the hidden state and cell state.

Example:
    Input:
        input_sequence = np.array([[1.0], [2.0], [3.0]])
        initial_hidden_state = np.zeros((1, 1))
        initial_cell_state = np.zeros((1, 1))

        lstm = LSTM(input_size=1, hidden_size=1)
        outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)

        print(final_h)
    Output:
        [[0.73698596]] (approximate)
Reasoning:
    The LSTM processes the input sequence [1.0, 2.0, 3.0] and produces the final hidden state [0.73698596].
"""


import numpy as np


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        h = initial_hidden_state
        cell = initial_cell_state
        for x_t in x:
            x_t = x_t.reshape(-1, 1)
            x_hidden_conc = np.vstack([h, x_t])
            f = _sigmoid(
                self.Wf @ x_hidden_conc
                + self.bf
            )
            i = _sigmoid(
                self.Wi @ x_hidden_conc
                + self.bi
            )


            c_tanh = np.tanh(
                self.Wc @ x_hidden_conc
                + self.bc
            )
            cell = f * cell + i * c_tanh

            o = _sigmoid(
                self.Wo @ x_hidden_conc + self.bo
            )
            h = o * np.tanh(cell)
        return o, h, cell
