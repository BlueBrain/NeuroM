# Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
# All rights reserved.
#
# This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the names of
#        its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from nose import tools as nt
from neurom.core import neuron
from neurom.core.tree import Tree
from neurom.exceptions import SomaError
from itertools import izip
import numpy as np

SOMA_A_PTS = [[11, 22, 33, 44, 1, 1, -1]]

SOMA_B_PTS = [
    [11, 22, 33, 44, 1, 1, -1],
    [11, 22, 33, 44, 2, 1, 1],
    [11, 22, 33, 44, 3, 1, 2]
]

SOMA_C_PTS_4 = [
    [11, 22, 33, 44, 1, 1, -1],
    [11, 22, 33, 44, 2, 1, 1],
    [11, 22, 33, 44, 3, 1, 2],
    [11, 22, 33, 44, 4, 1, 3]
]

SOMA_C_PTS_5 = [
    [11, 22, 33, 44, 1, 1, -1],
    [11, 22, 33, 44, 2, 1, 1],
    [11, 22, 33, 44, 3, 1, 2],
    [11, 22, 33, 44, 4, 1, 3],
    [11, 22, 33, 44, 5, 1, 4]
]


SOMA_C_PTS_6 = [
    [11, 22, 33, 44, 1, 1, -1],
    [11, 22, 33, 44, 2, 1, 1],
    [11, 22, 33, 44, 3, 1, 2],
    [11, 22, 33, 44, 4, 1, 3],
    [11, 22, 33, 44, 5, 1, 4],
    [11, 22, 33, 44, 6, 1, 5]
]


INVALID_PTS_0 = []

INVALID_PTS_2 = [

    [11, 22, 33, 44, 1, 1, -1],
    [11, 22, 33, 44, 2, 1, 1]
]

TREE = Tree([0.0, 0.0, 0.0, 1.0, 1, 1, 2] )
T1 = TREE.add_child(Tree([0.0, 1.0, 0.0, 1.0, 1, 1, 2]))
T2 = T1.add_child(Tree([0.0, 2.0, 0.0, 1.0, 1, 1, 2]))
T3 = T2.add_child(Tree([0.0, 4.0, 0.0, 2.0, 1, 1, 2]))
T4 = T3.add_child(Tree([0.0, 5.0, 0.0, 2.0, 1, 1, 2]))
T5 = T4.add_child(Tree([2.0, 5.0, 0.0, 1.0, 1, 1, 2]))
T6 = T4.add_child(Tree([0.0, 5.0, 2.0, 1.0, 1, 1, 2]))
T7 = T5.add_child(Tree([3.0, 5.0, 0.0, 0.75, 1, 1, 2]))
T8 = T7.add_child(Tree([4.0, 5.0, 0.0, 0.75, 1, 1, 2]))
T9 = T6.add_child(Tree([0.0, 5.0, 3.0, 0.75, 1, 1, 2]))
T10 = T9.add_child(Tree([0.0, 6.0, 3.0, 0.75, 1, 1, 2]))

def test_make_SomaA():
    soma = neuron.make_soma(SOMA_A_PTS)
    nt.ok_('SomaA' in str(soma))
    nt.ok_(isinstance(soma, neuron.SomaA))
    nt.assert_equal(soma.center, (11, 22, 33))
    nt.ok_(soma.radius == 44)

def test_make_SomaB():
    soma = neuron.make_soma(SOMA_B_PTS)
    nt.ok_('SomaB' in str(soma))
    nt.ok_(isinstance(soma, neuron.SomaB))
    nt.assert_equal(soma.center, (11, 22, 33))
    nt.ok_(soma.radius == 0.0)


def check_SomaC(points):
    soma = neuron.make_soma(points)
    nt.ok_('SomaC' in str(soma))
    nt.ok_(isinstance(soma, neuron.SomaC))
    nt.assert_equal(soma.center, (11, 22, 33))
    nt.ok_(soma.radius == 0.0)


def test_make_SomaC():
    check_SomaC(SOMA_C_PTS_4)
    check_SomaC(SOMA_C_PTS_5)
    check_SomaC(SOMA_C_PTS_6)


@nt.raises(SomaError)
def test_invalid_soma_points_0_raises_SomaError():
    neuron.make_soma(INVALID_PTS_0)


@nt.raises(SomaError)
def test_invalid_soma_points_2_raises_SomaError():
    neuron.make_soma(INVALID_PTS_2)


def test_neuron():
    soma = neuron.make_soma(SOMA_A_PTS)
    nrn = neuron.Neuron(soma, ['foo', 'bar'])
    nt.assert_equal(nrn.soma.center, (11, 22, 33))
    nt.assert_equal(nrn.neurites, ['foo', 'bar'])
    nt.assert_equal(nrn.name, 'Neuron')
    nrn = neuron.Neuron(soma, ['foo', 'bar'], 'test')
    nt.assert_equal(nrn.name, 'test')


def test_i_neurites_chains():
    soma = neuron.make_soma(SOMA_A_PTS)
    nrn = neuron.Neuron(soma, ['foo', 'bar', 'baz'])
    s = 'foobarbaz'
    for i, j in izip(s, nrn.i_neurites(iter)):
        nt.assert_equal(i, j)


def test_i_neurites_filter():
    soma = neuron.make_soma(SOMA_A_PTS)
    nrn = neuron.Neuron(soma, ['foo', 'bar', 'baz'])
    ref = 'barbaz'
    for i, j in izip(ref,
                     nrn.i_neurites(iter,
                                    tree_filter=lambda s: s.startswith('b'))):
        nt.assert_equal(i, j)


def test_bounding_box():

    soma = neuron.make_soma([[0, 0, 0, 1, 1, 1, -1]])
    nrn = neuron.Neuron(soma, [TREE])
    ref1 = ((-1, -1, -1), (4.0, 6.0, 3.0))

    for a, b in izip(nrn.bounding_box(), ref1):
        nt.assert_true(np.allclose(a, b))

    soma = neuron.make_soma(SOMA_A_PTS)
    nrn = neuron.Neuron(soma, [TREE])
    ref2 = ((-33, -22, -11), (55, 66, 77))

    for a, b in izip(nrn.bounding_box(), ref2):
        nt.assert_true(np.allclose(a, b))
