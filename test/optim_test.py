from pynn.optim import Optimizer, SGD

from pynn.layer import Layer
from pynn.nn import NeuralNetwork

import numpy as np
import pytest


def test_optimizer_step():

    opt = Optimizer()

    with pytest.raises(NotImplementedError):
        opt.step(NeuralNetwork([]))


def test_SGD_step():
    opt = SGD(1)

    layer = Layer()

    # Assign parameters and corresponding gradients associated to the layer
    layer.params = {"p": 5 * np.ones(10)}
    layer.grads = {"p": 4 * np.ones(10)}

    nn = NeuralNetwork([layer])

    opt.step(nn)

    # Extract modified (optimized) parameters from the neural network layer
    out = nn.layers[0].params["p"]

    assert np.allclose(out, np.ones(10))
