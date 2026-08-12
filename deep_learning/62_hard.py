"""
Implement a Simple RNN with Backpropagation Through Time (BPTT)
 - Hard
 - Deep Learning

Task: Implement a Simple RNN with Backpropagation Through Time (BPTT)
Your task is to implement a simple Recurrent Neural Network (RNN) and backpropagation
through time (BPTT) to learn from sequential data. The RNN will process input sequences,
update hidden states, and perform backpropagation to adjust weights based on the error gradient.

Write a class SimpleRNN with the following methods:
 - __init__(self, input_size, hidden_size, output_size): Initializes the RNN with random weights and zero biases.
 - forward(self, x): Processes a sequence of inputs and returns the hidden states and output.
 - backward(self, x, y, learning_rate): Performs backpropagation through time (BPTT) to adjust the weights based on the loss.

In this task, the RNN will be trained on sequence prediction, where the network
will learn to predict the next item in a sequence. You should use 1/2 * Mean Squared
Error (MSE) as the loss function and make sure to aggregate the losses at each
time step by summing.

Example:
    Input:
        import numpy as np
            input_sequence = np.array([[1.0], [2.0], [3.0], [4.0]])
            expected_output = np.array([[2.0], [3.0], [4.0], [5.0]])
            # Initialize RNN
            rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1)

            # Forward pass
            output = rnn.forward(input_sequence)

            # Backward pass
            rnn.backward(input_sequence, expected_output, learning_rate=0.01)

            print(output)

            # The output should show the RNN predictions for each step of the input sequence.
    Output:
        [[x1], [x2], [x3], [x4]]
Reasoning:
    The RNN processes the input sequence [1.0, 2.0, 3.0, 4.0] and predicts the
    next item in the sequence at each step.
"""


import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN with random weights and zero biases.
        """
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size)*0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size)*0.01
        self.W_hy = np.random.randn(output_size, hidden_size)*0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        self.y_pred = []
        self.h = []

    def forward(self, x):
        """
        Forward pass through the RNN for a given sequence of inputs.
        """
        self.y_pred = []
        self.h = []
        h = np.zeros((self.hidden_size, 1))
        for x_t in x:
            h = np.tanh(self.W_xh @ x_t.reshape((-1, 1)) + self.W_hh @ h + self.b_h)
            y_t = self.W_hy @ h + self.b_y
            self.h.append(h)
            self.y_pred.append(y_t)
        return self.y_pred

    def backward(self, x, y, learning_rate):
        """
        Backpropagation through time to adjust weights based on error gradient.
        """
        d_h_next = 0
        d_W_hy = 0
        d_b_y = 0
        d_W_xh = 0
        d_W_hh = 0
        d_b_h = 0
        for t in reversed(range(len(y))):
            # Compute the gradient of the loss with respect to the outputs
            d_L = self.y_pred[t] - y[t].reshape((-1, 1))

            # Compute the gradients for the output layer weights and biases
            d_W_hy += d_L @ self.h[t].T
            d_b_y += d_L

            # Backpropagate the gradients through the hidden layers
            d_h = self.W_hy.T @ d_L + d_h_next
            d_h_raw = d_h * (1 - self.h[t] ** 2)
            d_h_next = self.W_hh.T @ d_h_raw

            # Compute the gradients for the hidden layer weights and biases
            d_W_xh += d_h_raw @ x[t].reshape((-1, 1)).T
            if t - 1 < 0:
                d_W_hh += d_h_raw @ np.zeros((self.hidden_size, 1)).T
            else:
                d_W_hh += d_h_raw @ self.h[t - 1].T
            d_b_h += d_h_raw

        self.W_xh -= learning_rate * d_W_xh
        self.W_hh -= learning_rate * d_W_hh
        self.W_hy -= learning_rate * d_W_hy
        self.b_h -= learning_rate * d_b_h
        self.b_y -= learning_rate * d_b_y
