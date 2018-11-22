from pynn.layer import Layer, Linear, Activation, Tanh

import numpy as np
import pytest


def test_layer():

    L = Layer()

    with pytest.raises(NotImplementedError):
        L.forward(np.ones(10))

    with pytest.raises(NotImplementedError):
        L.backward(np.ones(10))


def test_linear_forward_1d():

    L = Linear(1, 1)

    assert L.params["w"].size == 1
    assert L.params["b"].size == 1

    L.params["w"] = 3
    L.params["b"] = -1

    assert L.forward(2) == 5


def test_linear_forward():

    L = Linear(3, 2)

    assert L.params["w"].size == 6
    assert L.params["w"].shape[0] == 3
    assert L.params["w"].shape[1] == 2
    assert L.params["b"].size == 2

    L.params["w"] = np.array([[3, 3], [2, 2], [1, 1]])
    L.params["b"] = np.array([-1, -2])

    inputs = np.ones(3)
    outputs = np.array([5, 4])

    assert np.allclose(L.forward(inputs), outputs)


def test_linear_backward_1d():
    pass


def test_linear_backward():
    pass


def test_linear_repr():
    L = Linear(3, 4)

    assert repr(L) == "Linear(3, 4)"


def test_activation_forward():
    A = Activation(np.sin, np.cos)

    x = np.linspace(0.0, 4.0 * np.pi, 100)
    y = np.sin(x)

    assert np.allclose(A.forward(x), y)


def test_activation_backward():
    A = Activation(np.sin, np.cos)

    x = np.linspace(0.0, 4.0 * np.pi, 100)
    dy = np.cos(x)
    grad = np.ones(x.size)

    # The forward pass is needed to store the inputs used by the backward pass
    A.forward(x)

    assert np.allclose(A.backward(grad), dy)


def test_tanh_forward():
    A = Tanh()

    x = np.linspace(-10, 10, 100)
    y = np.tanh(x)

    assert np.allclose(A.forward(x), y)


def test_tanh_backward():
    A = Tanh()

    x = np.linspace(-10, 10, 100)
    dy = 1.0 - np.tanh(x) ** 2
    grad = np.ones(x.size)

    # The forward pass is needed to store the inputs used by the backward pass
    A.forward(x)

    assert np.allclose(A.backward(grad), dy)


def test_tanh_repr():
    A = Tanh()

    assert repr(A) == "Tanh()"
