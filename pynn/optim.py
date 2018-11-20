"""
Optimizer.

An optimizers are used to adjust the parameters of the neural network during
training in order to minimise the loss.
"""

from pynn.nn import NeuralNetwork


class Optimizer:
    def step(self, nn: NeuralNetwork) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, nn: NeuralNetwork) -> None:
        for param, grad in nn.params_and_grads():
            param -= self.lr * grad
