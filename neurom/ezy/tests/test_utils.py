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

'''Test neurom.ezy.utils'''

import os
from copy import deepcopy
from neurom import ezy
from neurom.ezy import utils as ezy_utils
from collections import namedtuple
from neurom.core.types import NeuriteType
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
MORPH_FILE = os.path.join(DATA_PATH, 'swc', 'Neuron.swc')
NEURON = ezy.load_neuron(MORPH_FILE)


def test_eq():
    other = ezy.load_neuron(MORPH_FILE)
    nt.assert_true(ezy_utils.neurons_eq(NEURON, other))


def test_compare_neurites():

    fake_neuron = namedtuple('Neuron', 'neurites')
    fake_neuron.neurites = []
    nt.assert_false(ezy_utils._compare_neurites(NEURON, fake_neuron, NeuriteType.axon))
    nt.assert_true(fake_neuron, fake_neuron)

    neuron2 = deepcopy(NEURON)

    n_types = set([n.type for n in NEURON.neurites])

    for n_type in n_types:
        nt.assert_true(ezy_utils._compare_neurites(NEURON, neuron2, n_type))

    neuron2.neurites[1].children[0].value[1] += 0.01

    nt.assert_false(ezy_utils._compare_neurites(NEURON, neuron2, neuron2.neurites[1].type))
