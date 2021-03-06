"""
Layer of the neuronal network.

A neuronal networks is composed of multiple layers (input layer, hidden layers
and output layer).
Each layer has to propagate its inputs forward (forward pass) and propagate
the gradients backward (backward pass).
"""

from pynn.tensor import Tensor
from pynn.utils import tanh, tanh_derivative, softplus, sigmoid, relu, relu_derivative

from abc import ABC, abstractmethod
from typing import Dict, Callable

import numpy as np


class Layer(ABC):
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Compute the output of the layer corresponding to the given input, i.e.
        propagate the inputs forward through the layer.
        """
        pass

    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        """
        Propagate the gradient backward through the layer.
        """
        pass


class Linear(Layer):
    """
    Linear layer.

    A linear layer propagates the inputs forward via a linear function.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.params["w"] = np.random.rand(input_size, output_size)
        self.params["b"] = np.random.rand(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Propagate the inputs forward via a linear function:
            outputs = inputs @ w + b
        """
        # Save copy of inputs for backpropagation
        self.inputs = inputs

        return np.dot(inputs, self.params["w"]) + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        # Gradient with respect to w
        self.grads["w"] = np.dot(self.inputs.T, grad)

        # Gradient with respect to b
        self.grads["b"] = np.sum(grad, axis=0)

        # Gradient with respect to the input
        return np.dot(grad, self.params["w"].T)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.input_size}, {self.output_size})"


# Definition of a function taking a tensor and returning a tensor for type hints
F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
    Activation layer.

    An activation layer apply a (non-linear) function to its inputs element
    by element.
    """

    def __init__(self, f: F, df: F) -> None:
        super().__init__()

        self.f = f
        self.df = df

    def forward(self, inputs: Tensor) -> Tensor:
        # Save copy of inputs for backpropagation
        self.inputs = inputs

        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.df(self.inputs) * grad

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Tanh(Activation):
    """
    tanh activation function.


    Propagate the inputs forward via a tanh function:
        outputs = tanh(inputs)
    """

    def __init__(self):
        super().__init__(tanh, tanh_derivative)


class SoftPlus(Activation):
    """
    Softplus activation function.

    Propagate the inputs forward via a tanh function:
        outputs = softplus(inputs) = log(1 + exp(inputs))

    The softplus activation function is a smooth approximation of the ReLU activation function.
    """

    def __init__(self):
        super().__init__(softplus, sigmoid)


class ReLU(Activation):
    """
    ReLU activation function.

    Propagate the inputs forward via a rectified linear unit (ReLU):
        outputs = ReLU(inputs) = 0 if inputs < 0 else inputs
    """

    def __init__(self):
        super().__init__(relu, relu_derivative)
