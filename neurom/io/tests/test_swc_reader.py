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

import os

import numpy as np

from neurom import load_neuron
from neurom.core.dataformat import COLS
from neurom.io import swc

from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')
SWC_SOMA_PATH = os.path.join(SWC_PATH, 'soma')


def test_read_single_neurite():
    neuron = load_neuron(os.path.join(SWC_PATH, 'point_soma_single_neurite.swc'))
    nt.eq_(n.neurites[0].root_node, [1])
    nt.eq_(len(neuron.soma_points()), 1)
    nt.eq_(len(neuron.sections), 2)


def test_read_split_soma():
    neuron = load_neuron(os.path.join(SWC_PATH, 'split_soma_single_neurites.swc'))
    root_nodes = [neurite.root_node for neurite in neuron.neurites]
    nt.eq_(root_nodes, [1, 3])
    nt.eq_(len(neuron.soma_points()), 3)
    nt.eq_(len(neuron.sections), 4)

    ref_ids = [[-1, 0],
               [0, 1, 2, 3, 4],
               [0, 5, 6],
               [6, 7, 8, 9, 10],
               []]

    for s, r in zip(neuron.sections, ref_ids):
        nt.eq_(s.ids, r)


def test_simple_reversed():
    neuron = load_neuron(os.path.join(SWC_PATH, 'simple_reversed.swc'))
    root_nodes = [neurite.root_node for neurite in neuron.neurites]
    nt.eq_(root_nodes, [5, 6])
    nt.eq_(len(neuron.soma_points()), 1)
    nt.eq_(len(neuron.sections), 7)
