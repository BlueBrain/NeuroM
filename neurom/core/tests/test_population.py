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

from nose import tools as nt
from neurom.core.population import Population
from neurom import load_neuron

DATA_PATH = Path(__file__).parent.parent.parent.parent / 'test_data'

NRN1 = load_neuron(Path(DATA_PATH, 'swc/Neuron.swc'))
NRN2 = load_neuron(Path(DATA_PATH, 'swc/Single_basal.swc'))
NRN3 = load_neuron(Path(DATA_PATH, 'swc/Neuron_small_radius.swc'))

NEURONS = [NRN1, NRN2, NRN3]
TOT_NEURITES = sum(len(N.neurites) for N in NEURONS)
POP = Population(NEURONS, name='foo')


def test_population():
    nt.assert_equal(len(POP.neurons), 3)
    nt.ok_(POP.neurons[0].name, 'Neuron')
    nt.ok_(POP.neurons[1].name, 'Single_basal')
    nt.ok_(POP.neurons[2].name, 'Neuron_small_radius')

    nt.assert_equal(len(POP.somata), 3)

    nt.assert_equal(len(POP.neurites), TOT_NEURITES)

    nt.assert_equal(POP.name, 'foo')


def test_neurons():
    for i, n in enumerate(NEURONS):
        nt.assert_true(n is POP.neurons[i])


def test_iterate_neurons():
    for a, b in zip(NEURONS, POP):
        nt.assert_true(a is b)


def test_len():
    nt.assert_equal(len(POP), len(NEURONS))


def test_getitem():
    for i in range(len(NEURONS)):
        nt.assert_true(POP[i] is NEURONS[i])


def test_str():
    nt.ok_('Population' in str(POP))
