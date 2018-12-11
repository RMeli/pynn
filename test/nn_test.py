from pynn.layer import Linear, Tanh, ReLU
from pynn.nn import NeuralNetwork
from pynn.tensor import Tensor
from pynn.loss import MSE

import numpy as np

import pytest


def test_feed_forward_and():
    nn: NeuralNetwork = NeuralNetwork([Linear(2, 1)])

    nn.layers[0].params["w"] = np.array([[2], [2]])
    nn.layers[0].params["b"] = np.array([-3])

    assert nn.forward(np.array([1, 1]))[0] == pytest.approx(1)
    assert nn.forward(np.array([0, 1]))[0] == pytest.approx(-1)
    assert nn.forward(np.array([1, 0]))[0] == pytest.approx(-1)
    assert nn.forward(np.array([0, 0]))[0] == pytest.approx(-3)


def test_feed_forward_or():
    nn: NeuralNetwork = NeuralNetwork([Linear(2, 1)])

    nn.layers[0].params["w"] = np.array([[2], [2]])
    nn.layers[0].params["b"] = np.array([-1])

    assert nn.forward(np.array([1, 1]))[0] == pytest.approx(3)
    assert nn.forward(np.array([0, 1]))[0] == pytest.approx(1)
    assert nn.forward(np.array([1, 0]))[0] == pytest.approx(1)
    assert nn.forward(np.array([0, 0]))[0] == pytest.approx(-1)


def test_feed_forward_not_callable():
    nn: NeuralNetwork = NeuralNetwork([Linear(1, 1)])

    nn.layers[0].params["w"] = np.array([[-2]])
    nn.layers[0].params["b"] = np.array([1])

    assert nn(np.array([1]))[0] == pytest.approx(-1)
    assert nn(np.array([0]))[0] == pytest.approx(1)


def test_backpropagation_linear_tanh():
    """
    f(x) = tanh( w * x + b )
    f'(x) = tanh'(w * x + b) * w
    """

    nn: NeuralNetwork = NeuralNetwork([Linear(1, 1), Tanh()])

    nn.layers[0].params["w"] = np.array([[0.5]])
    nn.layers[0].params["b"] = np.array([-0.5])

    input: Tensor = 2 * np.ones(1)

    assert nn(input) == pytest.approx(np.tanh(0.5))

    grad: float = 1
    assert nn.backward(grad)[0] == pytest.approx((1 - np.tanh(0.5) ** 2) * 0.5)


def test_backpropagation_tanh_tanh():
    """
    f(x) = tanh( tanh(x) )
    f'(x) = tanh'( tanh(x) ) * tanh(x)
    """

    nn: NeuralNetwork = NeuralNetwork([Tanh(), Tanh()])

    input: Tensor = 0.5 * np.ones(1)

    assert nn(input)[0] == pytest.approx(np.tanh(np.tanh(0.5)))

    grad: float = 1
    assert nn.backward(grad)[0] == pytest.approx(
        (1 - np.tanh(np.tanh(0.5)) ** 2) * (1 - np.tanh(0.5) ** 2)
    )


def test_backpropagation_tanh_tanh_grad2():
    """
    f(x) = tanh( tanh(x) )
    f'(x) = tanh'( tanh(x) ) * tanh(x)
    """

    nn: NeuralNetwork = NeuralNetwork([Tanh(), Tanh()])

    input: Tensor = 0.5 * np.ones(1)

    assert nn(input)[0] == pytest.approx(np.tanh(np.tanh(0.5)))

    grad: float = 2
    assert nn.backward(grad)[0] == pytest.approx(
        grad * (1 - np.tanh(np.tanh(0.5)) ** 2) * (1 - np.tanh(0.5) ** 2)
    )


def test_repr():
    nn: NeuralNetwork = NeuralNetwork([Linear(10, 5), Tanh(), Linear(5, 2)])

    assert (
        repr(nn) == "NeuralNetwork([\n\tLinear(10, 5),\n\tTanh(),\n\tLinear(5, 2),\n])"
    )


def test_linear_tanh_linear():
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(1234)
    torch.set_printoptions(precision=10)

    net = nn.Sequential(
        nn.Linear(10, 5),
        nn.Tanh(),
        nn.Linear(5,2),
    )

    x = torch.rand(10)
    print(f"x = {x}")

    y = net(x)
    print(f"y = {y}")

    print(f"net[0].weight = {net[0].weight.data}")
    print(f"net[0].bias = {net[0].bias.data}")
    print(f"net[2].weight = {net[2].weight.data}")
    print(f"net[2].bias = {net[2].bias.data}")

    L = nn.MSELoss()

    loss = L(y, torch.tensor([0.123, 0.234]))
    print(loss.item())

    loss.backward()

    print(f"net[0].weight.grad = {net[0].weight.grad}")
    print(f"net[0].bias.grad = {net[0].bias.grad}")
    print(f"net[2].weight.grad = {net[2].weight.grad}")
    print(f"net[2].bias.grad = {net[2].bias.grad}")
    """

    # Define neural network
    nn: NeuralNetwork = NeuralNetwork([Linear(10, 5), Tanh(), Linear(5, 2)])

    # Define weight for first linear layer
    nn.layers[0].params["w"] = np.array(
        [
            [
                -0.2978996933,
                -0.0620447993,
                -0.1518878788,
                -0.0843434185,
                -0.2793551385,
                0.1268988550,
                -0.2834682167,
                -0.0201505125,
                0.1099246442,
                -0.1065928042,
            ],
            [
                0.1794327199,
                0.0398846865,
                0.1738307178,
                0.2028933465,
                -0.1395977736,
                0.1149241626,
                -0.1368159652,
                0.0991250277,
                -0.1652253270,
                0.1462771595,
            ],
            [
                0.0640188158,
                -0.1237535626,
                -0.1551083475,
                0.0818156302,
                0.2950474918,
                0.1517572105,
                -0.0305362642,
                -0.0153491795,
                0.1797402799,
                -0.2197806835,
            ],
            [
                0.1051295400,
                -0.1047832966,
                0.1829633415,
                -0.1128049493,
                0.0156366825,
                0.1067842245,
                0.2173210084,
                -0.0464802384,
                0.2884919941,
                -0.2675432563,
            ],
            [
                -0.0564081371,
                -0.3153346777,
                0.0261963010,
                0.0897391737,
                -0.1280111223,
                0.1313367486,
                -0.0512633622,
                -0.2747980654,
                0.2427785099,
                0.1949744523,
            ],
        ]
    ).transpose()

    # Define bias for first linear layer
    nn.layers[0].params["b"] = np.array(
        [0.1598871946, 0.2522428930, 0.1162832677, 0.1681351364, 0.2624163330]
    )

    # Define weight for second linear layer
    nn.layers[2].params["w"] = np.array(
        [
            [-0.0901057720, -0.3487843573, -0.2199362069, -0.0596988499, -0.0491428077],
            [-0.0030310154, 0.2562854886, 0.1434621215, -0.3306660652, -0.1343453825],
        ]
    ).transpose()

    # Define bias for second
    nn.layers[2].params["b"] = np.array([-0.1052068472, 0.2721802592])

    # Input (batch_size=1)
    x: Tensor = np.array(
        [
            [
                0.3186104298,
                0.2908077240,
                0.4196097851,
                0.3728144765,
                0.3768919110,
                0.0107794404,
                0.9454936385,
                0.7661116719,
                0.2634066939,
                0.1880336404,
            ]
        ]
    )

    # Compute prediction
    y = nn(x)

    # Check prediction value
    assert np.allclose(y, [-0.2401246876, 0.2529487908])

    # Loss function
    L = MSE()

    # Check loss value
    assert L.loss(y, np.array([0.123, 0.234])) == pytest.approx(0.066109299659729)

    # Compute loss gradient
    grad = L.grad(y, np.array([0.123, 0.234]))

    # Check there are no gradients before backpropagation
    assert nn.layers[0].grads.get("w", None) is None
    assert nn.layers[0].grads.get("b", None) is None
    assert nn.layers[2].grads.get("w", None) is None
    assert nn.layers[2].grads.get("b", None) is None

    # Backpropagation
    nn.backward(grad)

    t = np.array(
        [
            [
                0.0087200264,
                0.0079590958,
                0.0114842709,
                0.0102035329,
                0.0103151277,
                0.0002950217,
                0.0258771479,
                0.0209676567,
                0.0072091590,
                0.0051462795,
            ],
            [
                0.0371894054,
                0.0339441709,
                0.0489784293,
                0.0435163043,
                0.0439922400,
                0.0012582168,
                0.1103615686,
                0.0894234329,
                0.0307458173,
                0.0219479930,
            ],
            [
                0.0257711057,
                0.0235222578,
                0.0339405350,
                0.0301554520,
                0.0304852594,
                0.0008719052,
                0.0764771476,
                0.0619676672,
                0.0213059001,
                0.0152092790,
            ],
            [
                0.0041744434,
                0.0038101713,
                0.0054977397,
                0.0048846263,
                0.0049380488,
                0.0001412326,
                0.0123878857,
                0.0100376178,
                0.0034511623,
                0.0024636223,
            ],
            [
                0.0048741978,
                0.0044488637,
                0.0064193159,
                0.0057034274,
                0.0057658055,
                0.0001649071,
                0.0144644445,
                0.0117202057,
                0.0040296745,
                0.0028765949,
            ],
        ]
    ).transpose()
    assert np.allclose(nn.layers[0].grads["w"], t)

    t = np.array(
        [0.0273689292, 0.1167237535, 0.0808859468, 0.0131020295, 0.0152982995]
    ).transpose()
    assert np.allclose(nn.layers[0].grads["b"], t)

    t = np.array(
        [
            [0.1461823881, -0.1217547730, -0.0520500802, -0.1405923814, 0.0029136201],
            [-0.0076281782, 0.0063534812, 0.0027161089, 0.0073364768, -0.0001520403],
        ]
    ).transpose()
    assert np.allclose(nn.layers[2].grads["w"], t)

    t = np.array([-0.3631246984, 0.0189487934]).transpose()
    assert np.allclose(nn.layers[2].grads["b"], t)
