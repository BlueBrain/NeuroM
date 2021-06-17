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
from neurom.core.neuron import Neuron
from neurom import load_neuron

import pytest

DATA_PATH = Path(__file__).parent.parent / 'data'

FILES = [DATA_PATH / 'swc/Neuron.swc',
         DATA_PATH / 'swc/Single_basal.swc',
         DATA_PATH / 'swc/Neuron_small_radius.swc']

NEURONS = [load_neuron(f) for f in FILES]
TOT_NEURITES = sum(len(N.neurites) for N in NEURONS)
populations = [Population(NEURONS, name='foo'),
               Population(FILES, name='foo', cache=True)]


@pytest.mark.parametrize('pop', populations)
def test_names(pop):
    assert pop[0].name, 'Neuron'
    assert pop[1].name, 'Single_basal'
    assert pop[2].name, 'Neuron_small_radius'
    assert pop.name == 'foo'


def test_indexing():
    pop = populations[0]
    for i, n in enumerate(NEURONS):
        assert n is pop[i]
    with pytest.raises(ValueError, match='no 10 index'):
        pop[10]


def test_cache():
    pop = populations[1]
    for n in pop._files:
        assert isinstance(n, Neuron)


def test_double_indexing():
    pop = populations[0]
    for i, n in enumerate(NEURONS):
        assert n is pop[i]
    # second time to assure that generator is available again
    for i, n in enumerate(NEURONS):
        assert n is pop[i]


def test_iterating():
    pop = populations[0]
    for a, b in zip(NEURONS, pop):
        assert a is b

    for a, b in zip(NEURONS, pop.somata):
        assert a.soma is b


@pytest.mark.parametrize('pop', populations)
def test_len(pop):
    assert len(pop) == len(NEURONS)


def test_getitem():
    pop = populations[0]
    for i in range(len(NEURONS)):
        assert pop[i] is NEURONS[i]


@pytest.mark.parametrize('pop', populations)
def test_str(pop):
    assert 'Population' in str(pop)
