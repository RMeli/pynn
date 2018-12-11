from pynn.utils import (
    size,
    tanh,
    tanh_derivative,
    softplus,
    sigmoid,
    relu,
    relu_derivative,
)

import numpy as np
import pytest


def test_size_no_args():
    with pytest.raises(ValueError):
        n = size()


def test_size_error():
    x = np.ones(10)
    y = np.ones(10)
    z = np.ones(5)

    with pytest.raises(ValueError):
        n = size(x, y, z)


def test_size_one(n: int = 10):
    x = np.ones(n)

    assert size(x) == n


def test_size(n: int = 10):
    a = np.ones(n)
    b = np.ones(n)
    c = np.ones(n)
    d = np.ones(n)

    assert size(a, b, c, d) == n


def test_tanh():

    assert tanh(0) == pytest.approx(0)
    assert tanh(100) == pytest.approx(1)
    assert tanh(-100) == pytest.approx(-1)


def test__tanh_derivative():

    assert tanh_derivative(0) == pytest.approx(1)
    assert tanh_derivative(100) == pytest.approx(0)
    assert tanh_derivative(-100) == pytest.approx(0)


def test_softplus():
    assert softplus(0) == pytest.approx(np.log(2))
    assert softplus(-100) == pytest.approx(0)
    assert softplus(100) == pytest.approx(100)


def test_sigmoid():
    assert sigmoid(0) == pytest.approx(0.5)
    assert sigmoid(-100) == pytest.approx(0)
    assert sigmoid(100) == pytest.approx(1)


def test_relu():
    assert relu(-100) == pytest.approx(0)
    assert relu(-10) == pytest.approx(0)
    assert relu(-1) == pytest.approx(0)
    assert relu(0) == pytest.approx(0)
    assert relu(1) == pytest.approx(1)
    assert relu(10) == pytest.approx(10)
    assert relu(100) == pytest.approx(100)

    assert np.allclose(relu(np.array([-1, 0, 1, 10])), np.array([0, 0, 1, 10]))


def test_relu_derivative():
    assert relu_derivative(-100) == pytest.approx(0)
    assert relu_derivative(-10) == pytest.approx(0)
    assert relu_derivative(-1) == pytest.approx(0)
    assert relu_derivative(0) == pytest.approx(0)
    assert relu_derivative(1) == pytest.approx(1)
    assert relu_derivative(10) == pytest.approx(1)
    assert relu_derivative(100) == pytest.approx(1)

    assert np.allclose(
        relu_derivative(np.array([-1, 0, 1, 10])), np.array([0, 0, 1, 1])
    )
