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

'''Test neurom.ezy.Population'''

import os
from nose import tools as nt
from itertools import izip
from neurom.ezy.population import Population
from neurom.core.types import TreeType

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
VALID_DIR = os.path.join(DATA_PATH, 'valid_set')

def test_construct_population():
    pop = Population(VALID_DIR)
    nt.ok_(pop is not None)


class TestEzyPopulation(object):

    def setUp(self):
        self.directory = VALID_DIR
        self.pop = Population(VALID_DIR)
        self.n_somata = len(self.pop.somata)

    def test_iter_somata(self):
        sm_it = self.pop.iter_somata()
        for soma in self.pop.somata:
            nt.assert_almost_equal(sm_it.next().radius, soma.radius)

    def test_get_n_neurites(self):
        nrts_all = sum(len(neuron.neurites) for neuron in self.pop.neurons)
        nt.assert_equal(nrts_all, self.pop.get_n_neurites())

        nrts_axons = sum(nrn.get_n_neurites(neurite_type=TreeType.axon) for nrn in self.pop.neurons)
        nt.assert_equal(nrts_axons, self.pop.get_n_neurites(neurite_type=TreeType.axon))


    def test_iter_neurites(self):
        nrts_it = self.pop.iter_neurites()
        nt.assert_equal(len(self.pop.neurites), len(list(nrts_it)))

    def test_iter_neurons(self):
        nrns_it = self.pop.iter_neurons()
        nt.assert_equal(len(self.pop.neurons), len(list(nrns_it)))
