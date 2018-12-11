from pynn.optim import SGD

from pynn.layer import Linear
from pynn.nn import NeuralNetwork

import numpy as np


def test_SGD_step():
    opt = SGD(1)

    # Dummy linear layer
    layer = Linear(0, 0)

    # Assign parameters and corresponding gradients associated to the layer
    layer.params = {"p": 5 * np.ones(10)}
    layer.grads = {"p": 4 * np.ones(10)}

    nn = NeuralNetwork([layer])

    opt.step(nn)

    # Extract modified (optimized) parameters from the neural network layer
    out = nn.layers[0].params["p"]

    assert np.allclose(out, np.ones(10))
