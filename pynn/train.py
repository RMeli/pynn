"""
Train neuronal network using training data.
"""

from pynn.tensor import Tensor
from pynn.nn import NeuralNetwork
from pynn.loss import Loss, MSE
from pynn.optim import Optimizer, SGD
from pynn.data import DataIterator, BatchIterator


def train(
    nn: NeuralNetwork,
    inputs: Tensor,
    targets: Tensor,
    num_epochs: int = 5000,
    iterator: DataIterator = BatchIterator(),
    loss: Loss = MSE(),
    optimizer: Optimizer = SGD(),
) -> None:
    """
    Train neuronal network on the given inputs and targets (training set).
    """

    for epoch in range(num_epochs):
        epoch_loss: float = 0

        # Iterate over data in batches
        for batch in iterator(inputs, targets):
            # Forward propagation (prediction)
            predicted = nn.forward(batch.inputs)

            # Computation of loss
            epoch_loss += loss.loss(predicted, batch.targets)

            # Backpropagation
            grad = loss.grad(predicted, batch.targets)
            nn.backward(grad)

            # Change neural network parameters to reduce the loss
            optimizer.step(nn)

        if epoch % (num_epochs / 10) == 0:
            print("\n========== Epoch", epoch, "==========")
            print("Loss: ", epoch_loss)
