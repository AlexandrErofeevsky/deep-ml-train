"""
Implementing Basic Autograd Operations
 - Medium
 - Deep Learning

Inspired by Andrej Karpathy's micrograd â check out his excellent video:
https://youtu.be/VMj-3S1tku0

Implement a Value class that wraps a scalar number and supports automatic differentiation.
Your class should:
 1. Support + and * operations that work with other Value objects or plain numbers,
    returning a new Value with the correct result.

 2. Support a relu() method that applies the ReLU activation function
    (outputs the value if positive, 0 otherwise).

 3. Support a backward() method that computes gradients for all upstream
    Value objects in the computation graph using backpropagation.

Each operation should track its inputs and know how to locally compute its gradient contribution.
The backward() method should process nodes in the correct order so that gradients accumulate properly from output back to inputs.

Hints:
Each operation creates a new Value that remembers which values produced it (_children).
Think about what order nodes need to be processed during the backward pass.
Gradients should be accumulated (+=), not replaced.

Example:
    Input:
        a = Value(2)
        b = Value(-3)
        c = Value(10)
        d = a + b * c
        e = d.relu()
        e.backward()
        print(a, b, c, d, e)
    Output:
        Value(data=2, grad=0)
        Value(data=-3, grad=0)
        Value(data=10, grad=0)
Reasoning:
    The output reflects the forward computation and gradients after backpropagation.
    The ReLU on 'd' zeros out its output and gradient due to the negative data value.
"""


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        result = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = _backward

        return result

    def __mul__(self, other):
        result = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = _backward

        return result

    def relu(self):
        result = Value(max(0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (self.data > 0) * result.grad

        result._backward = _backward

        return result

    def backward(self):
        topo = []

        def build_topo(v):
            for child in v._prev:
                build_topo(child)
            topo.append(v)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()
