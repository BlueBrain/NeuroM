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

'''Test neurom.ezy.Neuron'''

import os
import numpy as np
from neurom import ezy
from nose import tools as nt
from neurom.analysis.morphtree import i_section_length, i_segment_length

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')

def test_construct_neuron():
    filename = os.path.join(SWC_PATH, 'Neuron.swc')
    nrn = ezy.Neuron(filename)

    nt.ok_(nrn._nrn is not None)


class TestEzyNeuron(object):

    def setUp(self):
        self.filename = os.path.join(SWC_PATH, 'Neuron.swc')
        self.neuron = ezy.Neuron(self.filename)
        self.seclen = []
        for t in self.neuron._nrn.neurite_trees:
            self.seclen.extend(ll for ll in i_section_length(t))

        self.seglen = []
        for t in self.neuron._nrn.neurite_trees:
            self.seglen.extend(ll for ll in i_segment_length(t))

    def test_get_section_lengths(self):
        seclen = self.neuron.get_section_lengths()
        nt.assert_equal(len(seclen), 84)
        nt.assert_equal(seclen, self.seclen)

    def test_get_segment_lengths(self):
        seglen = self.neuron.get_segment_lengths()
        nt.assert_equal(len(seglen), 840)
        nt.assert_equal(seglen, self.seglen)

    def test_get_n_sections(self):
        nt.assert_equal(self.neuron.get_n_sections(), 84)

    def test_get_n_sections_per_neurite(self):
        nsecs = self.neuron.get_n_sections_per_neurite()
        nt.assert_equal(len(nsecs), 4)
        for ns in nsecs:
            nt.assert_equal(ns, 21)

    def test_get_n_neurites(self):
        nt.assert_equal(self.neuron.get_n_neurites(), 4)
