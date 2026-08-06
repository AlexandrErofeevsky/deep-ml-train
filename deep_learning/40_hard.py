"""
Implementing a Custom Dense Layer in Python
 - Hard
 - Deep Learning

Implementing a Custom Dense Layer in Python
You are provided with a base Layer class that defines the structure
of a neural network layer. Your task is to implement a subclass called Dense,
which represents a fully connected neural network layer. The Dense class should
extend the Layer class and implement the following methods:

    1. Initialization (__init__):
         - Define the layer with a specified number of neurons (n_units) and
        an optional input shape (input_shape).
         - Set up placeholders for the layer's weights (W), biases (w0), and optimizers.

    2. Weight Initialization (initialize):
         - Initialize the weights W using a uniform distribution with a limit of
        1 / sqrt(input_shape[0]), and bias w0 should be set to zero.
         - Initialize optimizers for W and w0.

    3. Parameter Count (parameters):
         - Return the total number of trainable parameters in the layer,
         which includes the parameters in W and w0.

    4. Forward Pass (forward_pass):
         - Compute the output of the layer by performing a dot product between
         the input X and the weight matrix W, and then adding the bias w0.

    5. Backward Pass (backward_pass):
         - Calculate and return the gradient with respect to the input.
         - If the layer is trainable, update the weights and biases using
         the optimizer's update rule.

    6. Output Shape (output_shape):
         - Return the shape of the output produced by the forward pass,
        which should be (self.n_units,).

Objective:
    Extend the Layer class by implementing the Dense class to ensure
    it functions correctly within a neural network framework.

Example:
    Input:
        # Initialize a Dense layer with 3 neurons and input shape (2,)
        dense_layer = Dense(n_units=3, input_shape=(2,))

        # Define a mock optimizer with a simple update rule
        class MockOptimizer:
            def update(self, weights, grad):
                return weights - 0.01 * grad

        optimizer = MockOptimizer()

        # Initialize the Dense layer with the mock optimizer
        dense_layer.initialize(optimizer)

        # Perform a forward pass with sample input data
        X = np.array([[1, 2]])
        output = dense_layer.forward_pass(X)
        print("Forward pass output:", output)

        # Perform a backward pass with sample gradient
        accum_grad = np.array([[0.1, 0.2, 0.3]])
        back_output = dense_layer.backward_pass(accum_grad)
        print("Backward pass output:", back_output)
    Output:
        Forward pass output: [[-0.00655782  0.01429615  0.00905812]]
        Backward pass output: [[ 0.00129588  0.00953634]]
Reasoning:
    The code initializes a Dense layer with 3 neurons and input shape (2,).
    It then performs a forward pass with sample input data and a backward
    pass with sample gradients.
    The output demonstrates the forward and backward pass results.
"""

import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)


# DO NOT CHANGE LAYER CLASS
class Layer(object):

    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

        self.W_optimizer = None
        self.w0_optimizer = None

    def initialize(self, optimizer: object):
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(
            -limit, limit, (self.input_shape[0], self.n_units),
        )
        self.w0 = np.zeros(self.n_units)
        self.W_optimizer = copy.copy(optimizer)
        self.w0_optimizer = copy.copy(optimizer)

    def parameters(self):
        return self.W.size + self.w0.size

    def forward_pass(self, X, training=None):
        self.layer_input = X
        return X @ self.W + self.w0

    def backward_pass(self, accum_grad):
        grad_input = accum_grad @ self.W.T
        grad_W = self.layer_input.T @ accum_grad
        grad_b = np.sum(accum_grad, axis=0)

        if self.trainable:
            self.W = self.W_optimizer.update(self.W, grad_W)
            self.w0 = self.w0_optimizer.update(self.w0, grad_b)

        return grad_input

    def output_shape(self):
        return (self.n_units,)
