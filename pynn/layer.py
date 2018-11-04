"""
Layer of the neuronal network.

A neuronal networks is composed of multiple layers (imput layer, hidden layers
and output layer).
Each layer has to propagate its inputs forward (forward pass) and propagate
the gradients backward (backward pass).
"""

from pynn.tensor import Tensor

import numpy as np

class Layer():

    def __init__(self) -> None:
        pass

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Compute the output of the layer corresponding to the given input, i.e.
        propagate the inputs forward through the layer.
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Propagate the gradient backward through the layer.
        """
        raise NotImplementedError