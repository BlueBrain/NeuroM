from nose import tools as nt
from neurom.core import neuron

SOMA_A_PTS = [[1, 1, 11, 22, 33, 44, -1]]

SOMA_B_PTS = [
    [1, 1, 11, 22, 33, 44, -1],
    [2, 1, 11, 22, 33, 44, 1],
    [3, 1, 11, 22, 33, 44, 2]
]

SOMA_C_PTS_4 = [
    [1, 1, 11, 22, 33, 44, -1],
    [2, 1, 11, 22, 33, 44, 1],
    [3, 1, 11, 22, 33, 44, 2],
    [4, 1, 11, 22, 33, 44, 3]
]

SOMA_C_PTS_5 = [
    [1, 1, 11, 22, 33, 44, -1],
    [2, 1, 11, 22, 33, 44, 1],
    [3, 1, 11, 22, 33, 44, 2],
    [4, 1, 11, 22, 33, 44, 3],
    [5, 1, 11, 22, 33, 44, 4]
]


SOMA_C_PTS_6 = [
    [1, 1, 11, 22, 33, 44, -1],
    [2, 1, 11, 22, 33, 44, 1],
    [3, 1, 11, 22, 33, 44, 2],
    [4, 1, 11, 22, 33, 44, 3],
    [5, 1, 11, 22, 33, 44, 4],
    [6, 1, 11, 22, 33, 44, 5]
]


INVALID_PTS_0 = []

INVALID_PTS_2 = [

    [1, 1, 11, 22, 33, 44, -1],
    [2, 1, 11, 22, 33, 44, 1]
]

def test_make_SomaA():
    soma = neuron.make_soma(SOMA_A_PTS)
    nt.ok_(isinstance(soma, neuron.SomaA))
    nt.assert_equal(soma.center, (11, 22, 33))
    nt.ok_(soma.radius == 44)

def test_make_SomaB():
    soma = neuron.make_soma(SOMA_B_PTS)
    nt.ok_(isinstance(soma, neuron.SomaB))
    nt.assert_equal(soma.center, (11, 22, 33))
    nt.ok_(soma.radius == 0.0)


def check_SomaC(points):
    soma = neuron.make_soma(points)
    nt.ok_(isinstance(soma, neuron.SomaC))
    nt.assert_equal(soma.center, (11, 22, 33))
    nt.ok_(soma.radius == 0.0)


def test_make_SomaC():
    check_SomaC(SOMA_C_PTS_4)
    check_SomaC(SOMA_C_PTS_5)
    check_SomaC(SOMA_C_PTS_6)


@nt.raises(Exception)
def test_invalid_soma_points_0_raises():
    neuron.make_soma(INVALID_PTS_0)


@nt.raises(Exception)
def test_invalid_soma_points_2_raises():
    neuron.make_soma(INVALID_PTS_2)


def test_neuron():
    nrn = neuron.Neuron(SOMA_A_PTS, ['foo', 'bar'])
    nt.assert_equal(nrn.soma.center, (11, 22, 33))
    nt.assert_equal(nrn.neurite_trees, ['foo', 'bar'])
