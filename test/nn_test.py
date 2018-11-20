from pynn.layer import Linear
from pynn.nn import NeuralNetwork

import numpy as np

import pytest

def test_feed_forward_and():
    nn = NeuralNetwork([Linear(2, 1)])

    nn.layers[0].params['w'] = np.array([[2], [2]])
    nn.layers[0].params['b'] = np.array([-3])

    assert nn.forward(np.array([1, 1]))[0] == pytest.approx(1)
    assert nn.forward(np.array([0, 1]))[0] == pytest.approx(-1)
    assert nn.forward(np.array([1, 0]))[0] == pytest.approx(-1)
    assert nn.forward(np.array([0, 0]))[0] == pytest.approx(-3)


def test_feed_forward_or():
    nn = NeuralNetwork([Linear(2, 1)])

    nn.layers[0].params['w'] = np.array([[2], [2]])
    nn.layers[0].params['b'] = np.array([-1])

    assert nn.forward(np.array([1, 1]))[0] == pytest.approx(3)
    assert nn.forward(np.array([0, 1]))[0] == pytest.approx(1)
    assert nn.forward(np.array([1, 0]))[0] == pytest.approx(1)
    assert nn.forward(np.array([0, 0]))[0] == pytest.approx(-1)


def test_feed_forward_not_callable():
    nn = NeuralNetwork([Linear(1, 1)])

    nn.layers[0].params['w'] = np.array([[-2]])
    nn.layers[0].params['b'] = np.array([1])

    assert nn(np.array([1]))[0] == pytest.approx(-1)
    assert nn(np.array([0]))[0] == pytest.approx(1)



