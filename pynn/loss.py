"""
Loss functions.

A loss function measure how good the predictions of the model are, compared to 
the expected results.
The better the model (after training), the lower the value of the loss function.
During training, the parameters in the model are modified in order to lower the 
value of the loss function.
"""

from pynn.tensor import Tensor

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
