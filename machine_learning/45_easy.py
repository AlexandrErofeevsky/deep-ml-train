"""
Linear Kernel Function
 - Easy
 - Machine Learning

Write a Python function kernel_function that computes the linear kernel between
two input vectors x1 and x2. The linear kernel is defined as the dot product
(inner product) of two vectors.

Example:
    Input:
        import numpy as np

        x1 = np.array([1, 2, 3])
        x2 = np.array([4, 5, 6])

        result = kernel_function(x1, x2)
        print(result)
    Output:
        32
Reasoning:
    The linear kernel between x1 and x2 is computed as:1*4 + 2*5 + 3*6 = 32
"""


import numpy as np

def kernel_function(x1, x2):
	return x1 @ x2
