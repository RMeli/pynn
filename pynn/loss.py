"""
Loss functions.

A loss function measure how good the predictions of the model are, compared to 
the expected results.
The better the model (after training), the lower the value of the loss function.
During training, the parameters in the model are modified in order to lower the 
value of the loss function.
"""

from pynn.tensor import Tensor
from pynn.utils import size

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def loss(self, predicted: Tensor, target: Tensor) -> float:
        """
        Loss function.

        Computes the loss given the predicted and target values.
        """
        pass

    @abstractmethod
    def grad(self, predicted: Tensor, target: Tensor) -> Tensor:
        """
        Gradient of the loss function.

        Computes the gradient of the loss function given the predicted and
        target values. 
        """
        pass


class MSE(Loss):
    """
    Means Square Error (MSE) loss function.
    """

    def loss(self, predicted: Tensor, target: Tensor) -> float:
        n = size(predicted, target)

        return np.sum((predicted - target) ** 2) / n

    def grad(self, predicted: Tensor, target: Tensor) -> float:

        return 2 * (predicted - target)
