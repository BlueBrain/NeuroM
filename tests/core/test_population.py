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

from pathlib import Path

from neurom.core.population import Population
from neurom import load_neuron

import pytest

DATA_PATH = Path(__file__).parent.parent / 'data'

NRN1 = load_neuron(DATA_PATH / 'swc/Neuron.swc')
NRN2 = load_neuron(DATA_PATH / 'swc/Single_basal.swc')
NRN3 = load_neuron(DATA_PATH / 'swc/Neuron_small_radius.swc')

NEURONS = [NRN1, NRN2, NRN3]
TOT_NEURITES = sum(len(N.neurites) for N in NEURONS)
POP = Population(NEURONS, name='foo')


def test_names():
    assert POP[0].name, 'Neuron'
    assert POP[1].name, 'Single_basal'
    assert POP[2].name, 'Neuron_small_radius'
    assert POP.name == 'foo'


def test_indexing():
    for i, n in enumerate(NEURONS):
        assert n is POP[i]
    with pytest.raises(ValueError, match='no 10 index'):
        POP[10]

def test_double_indexing():
    for i, n in enumerate(NEURONS):
        assert n is POP[i]
    # second time to assure that generator is available again
    for i, n in enumerate(NEURONS):
        assert n is POP[i]


def test_iterating():
    for a, b in zip(NEURONS, POP):
        assert a is b

    for a, b in zip(NEURONS, POP.somata):
        assert a.soma is b


def test_len():
    assert len(POP) == len(NEURONS)


def test_getitem():
    for i in range(len(NEURONS)):
        assert POP[i] is NEURONS[i]


def test_str():
    assert 'Population' in str(POP)
