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
from neurom.exceptions import SomaError
from itertools import izip

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


@nt.raises(SomaError)
def test_invalid_soma_points_0_raises_SomaError():
    neuron.make_soma(INVALID_PTS_0)


@nt.raises(SomaError)
def test_invalid_soma_points_2_raises_SomaError():
    neuron.make_soma(INVALID_PTS_2)


def test_neuron():
    nrn = neuron.Neuron(SOMA_A_PTS, ['foo', 'bar'])
    nt.assert_equal(nrn.soma.center, (11, 22, 33))
    nt.assert_equal(nrn.neurites, ['foo', 'bar'])
    nt.assert_equal(nrn.id, 'Neuron')
    nrn = neuron.Neuron(SOMA_A_PTS, ['foo', 'bar'], 'test')
    nt.assert_equal(nrn.id, 'test')


def test_i_neurites_chains():
    nrn = neuron.Neuron(SOMA_A_PTS, ['foo', 'bar', 'baz'])
    s = 'foobarbaz'
    for i, j in izip(s, nrn.i_neurites(iter)):
        nt.assert_equal(i, j)


def test_i_neurites_filter():
    nrn = neuron.Neuron(SOMA_A_PTS, ['foo', 'bar', 'baz'])
    ref = 'barbaz'
    for i, j in izip(ref,
                     nrn.i_neurites(iter,
                                    tree_filter=lambda s: s.startswith('b'))):
        nt.assert_equal(i, j)


@nt.raises(SomaError)
def test_neuron_invalid_soma_points_0_raises_SomaError():
    neuron.Neuron(INVALID_PTS_0, [1, 2, 3])


@nt.raises(SomaError)
def test_neuron_invalid_soma_points_2_raises_SomaError():
    neuron.Neuron(INVALID_PTS_2, [1, 2, 3])
