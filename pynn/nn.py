"""
Neural network.

A neural network is a sequential connection of layers. It has an input layer,
a number of hidden layers and an output layer.
"""

from pynn.tensor import Tensor
from pynn.layer import Layer

from typing import Sequence

class NeuralNetwork:

    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    
