from pynn.layer import Linear, Tanh
from pynn.nn import NeuralNetwork

import numpy as np

import pytest

def test_feed_forward_and():
    nn: NeuralNetwork = NeuralNetwork([Linear(2, 1)])

    nn.layers[0].params['w'] = np.array([[2], [2]])
    nn.layers[0].params['b'] = np.array([-3])

    assert nn.forward(np.array([1, 1]))[0] == pytest.approx(1)
    assert nn.forward(np.array([0, 1]))[0] == pytest.approx(-1)
    assert nn.forward(np.array([1, 0]))[0] == pytest.approx(-1)
    assert nn.forward(np.array([0, 0]))[0] == pytest.approx(-3)


def test_feed_forward_or():
    nn: NeuralNetwork = NeuralNetwork([Linear(2, 1)])

    nn.layers[0].params['w'] = np.array([[2], [2]])
    nn.layers[0].params['b'] = np.array([-1])

    assert nn.forward(np.array([1, 1]))[0] == pytest.approx(3)
    assert nn.forward(np.array([0, 1]))[0] == pytest.approx(1)
    assert nn.forward(np.array([1, 0]))[0] == pytest.approx(1)
    assert nn.forward(np.array([0, 0]))[0] == pytest.approx(-1)


def test_feed_forward_not_callable():
    nn: NeuralNetwork = NeuralNetwork([Linear(1, 1)])

    nn.layers[0].params['w'] = np.array([[-2]])
    nn.layers[0].params['b'] = np.array([1])

    assert nn(np.array([1]))[0] == pytest.approx(-1)
    assert nn(np.array([0]))[0] == pytest.approx(1)

def test_backpropagation_linear_tanh():
    """
    f(x) = tanh( w * x + b )
    f'(x) = tanh'(w * x + b) * w
    """

    nn: NeuralNetwork = NeuralNetwork([Linear(1, 1), Tanh()])

    nn.layers[0].params['w'] = np.array([[0.5]])
    nn.layers[0].params['b'] = np.array([-0.5])

    input: Tensor = 2 * np.ones(1)

    assert nn(input) == pytest.approx(np.tanh(0.5))

    grad: float = 1
    assert nn.backward(grad)[0] == pytest.approx((1 - np.tanh(0.5)**2) * 0.5)

def test_backpropagation_tanh_tanh():
    """
    f(x) = tanh( tanh(x) )
    f'(x) = tanh'( tanh(x) ) * tanh(x)
    """

    nn: NeuralNetwork = NeuralNetwork([Tanh(), Tanh()])

    input: Tensor = 0.5 * np.ones(1)

    assert nn(input)[0] == pytest.approx(np.tanh(np.tanh(0.5)))

    grad: float = 1
    assert nn.backward(grad)[0] == pytest.approx((1 - np.tanh(np.tanh(0.5))**2) * (1 - np.tanh(0.5)**2))

def test_backpropagation_tanh_tanh_grad():
    """
    f(x) = tanh( tanh(x) )
    f'(x) = tanh'( tanh(x) ) * tanh(x)
    """

    nn: NeuralNetwork = NeuralNetwork([Tanh(), Tanh()])

    input: Tensor = 0.5 * np.ones(1)

    assert nn(input)[0] == pytest.approx(np.tanh(np.tanh(0.5)))

    grad: float = 2
    assert nn.backward(grad)[0] == pytest.approx(grad * (1 - np.tanh(np.tanh(0.5))**2) * (1 - np.tanh(0.5)**2))

