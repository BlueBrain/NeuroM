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

import math
from pathlib import Path

import neurom as nm
from neurom.core import Neurite

from numpy.testing import assert_almost_equal

SWC_PATH = Path(__file__).parent.parent / 'data/swc/'
nrn = nm.load_neuron(SWC_PATH / 'point_soma_single_neurite.swc')

ROOT_NODE = nrn.neurites[0].morphio_root_node
RADIUS = .5
REF_LEN = 3


def test_init():
    nrt = Neurite(ROOT_NODE)
    assert nrt.type == nm.NeuriteType.basal_dendrite
    assert len(nrt.points) == 4


def test_neurite_length():
    nrt = Neurite(ROOT_NODE)
    assert_almost_equal(nrt.length, REF_LEN)


def test_neurite_area():
    nrt = Neurite(ROOT_NODE)
    area = 2 * math.pi * RADIUS * REF_LEN
    assert_almost_equal(nrt.area, area)


def test_neurite_volume():
    nrt = Neurite(ROOT_NODE)
    volume = math.pi * RADIUS * RADIUS * REF_LEN
    assert_almost_equal(nrt.volume, volume)


def test_str():
    nrt = Neurite(ROOT_NODE)
    assert 'Neurite' in str(nrt)


def test_neurite_hash():
    nrt = Neurite(ROOT_NODE)
    assert hash(nrt) == hash((nrt.type, nrt.root_node))
