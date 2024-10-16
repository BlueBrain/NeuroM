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
from neurom.core.morphology import Morphology
from neurom import load_morphology

import pytest

DATA_PATH = Path(__file__).parent.parent / 'data'

FILES = [
    DATA_PATH / 'swc/Neuron.swc',
    DATA_PATH / 'swc/Single_basal.swc',
    DATA_PATH / 'swc/Neuron_small_radius.swc',
]

NEURONS = [load_morphology(f) for f in FILES]
TOT_NEURITES = sum(len(N.neurites) for N in NEURONS)
populations = [Population(NEURONS, name='foo'), Population(FILES, name='foo', cache=True)]


@pytest.mark.parametrize('pop', populations)
def test_names(pop):
    assert pop[0].name, 'Morphology'
    assert pop[1].name, 'Single_basal'
    assert pop[2].name, 'Neuron_small_radius'
    assert pop.name == 'foo'


def test_indexing():
    pop = populations[0]
    for i, n in enumerate(NEURONS):
        assert n.name == pop[i].name
        assert (n.points == pop[i].points).all()
    with pytest.raises(ValueError, match='no 10 index'):
        pop[10]


def test_cache():
    pop = populations[1]
    for n in pop._files:
        assert isinstance(n, Morphology)


@pytest.mark.parametrize("cache", [True, False])
def test_reset_cache(cache):
    pop = Population(FILES, cache=cache, process_subtrees=True)

    assert pop._process_subtrees is True
    for n in pop:
        assert isinstance(n, Morphology)
        assert n.process_subtrees is True

    pop.process_subtrees = False
    assert pop._process_subtrees is False
    for n in pop:
        assert isinstance(n, Morphology)
        assert n.process_subtrees is False

    mixed_pop = Population(FILES + NEURONS, cache=cache, process_subtrees=True)
    assert mixed_pop._process_subtrees is True
    for n in mixed_pop:
        assert isinstance(n, Morphology)
        assert n.process_subtrees is True

    mixed_pop.process_subtrees = False
    assert mixed_pop._process_subtrees is False
    for n in mixed_pop:
        assert isinstance(n, Morphology)
        assert n.process_subtrees is False


def test_double_indexing():
    pop = populations[0]
    for i, n in enumerate(NEURONS):
        assert n.name == pop[i].name
        assert (n.points == pop[i].points).all()
    # second time to assure that generator is available again
    for i, n in enumerate(NEURONS):
        assert n.name == pop[i].name
        assert (n.points == pop[i].points).all()


def test_iterating():
    pop = populations[0]
    for a, b in zip(NEURONS, pop):
        assert a.name == b.name
        assert (a.points == b.points).all()

    for a, b in zip(NEURONS, pop.somata):
        assert (a.soma.points == b.points).all()


@pytest.mark.parametrize('pop', populations)
def test_len(pop):
    assert len(pop) == len(NEURONS)


def test_getitem():
    pop = populations[0]
    for i in range(len(NEURONS)):
        assert pop[i].name == NEURONS[i].name
        assert (pop[i].points == NEURONS[i].points).all()


@pytest.mark.parametrize('pop', populations)
def test_str(pop):
    assert 'Population' in str(pop)
