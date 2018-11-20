"""
Neural network.

A neural network is a sequential connection of layers. It has an input layer,
a number of hidden layers and an output layer.
"""

from pynn.tensor import Tensor
from pynn.layer import Layer

from typing import Sequence, Iterator, Tuple


class NeuralNetwork:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def __call__(self, inputs: Tensor) -> Tensor:
        """
        Make a NeutalNetwork object callable for forward pass.
        """
        return self.forward(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagation.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]

                yield param, grad
