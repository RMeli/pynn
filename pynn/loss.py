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

import numpy as np

class Loss:

    def loss(self, predicted: Tensor, exact: Tensor) -> float:
        """
        Loss function.

        Computes the loss given the predicted and exact values.
        """
        
        raise NotImplementedError

    def grad(self, predicted: Tensor, exact: Tensor) -> Tensor:
        """
        Gradient of the loss function.

        Computes the graduent of the loss function given the predicted and
        exact values. 
        """

        raise NotImplementedError

class MSE(Loss):
    """
    Means Square Error (MSE) loss function.
    """

    def loss(self, predicted: Tensor, exact: Tensor) -> float:
        n = size(predicted, exact)

        return np.sum((predicted - exact)**2) / n

    def grad(self, predicted: Tensor, exact: Tensor) -> float:
        size(predicted, exact)

        return 2 * (predicted - exact)
