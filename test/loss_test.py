from pynn.loss import Loss, MSE

import numpy as np
import pytest


def test_mse_loss(n: int = 10):
    L = MSE()

    p = 2 * np.ones(n)
    e = np.ones(n)

    l = L.loss(p, e)

    assert l == pytest.approx(1)


def test_mse_grad(n: int = 10):
    L = MSE()

    p = 2 * np.ones(n)
    e = np.ones(n)

    g = L.grad(p, e)

    assert np.allclose(g, 2 * np.ones(n))
