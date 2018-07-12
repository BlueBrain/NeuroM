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

import numpy as np
from nose import tools as nt

import neurom as nm
from neurom.core import Neurite, Section

RADIUS = 4.
POINTS0 = np.array([[0., 0., 0., RADIUS],
                    [0., 0., 1., RADIUS],
                    [0., 0., 2., RADIUS],
                    [0., 0., 3., RADIUS],
                    [1., 0., 3., RADIUS],
                    [2., 0., 3., RADIUS],
                    [3., 0., 3., RADIUS]])

POINTS1 = np.array([[3., 0., 3., RADIUS],
                    [3., 0., 4., RADIUS],
                    [3., 0., 5., RADIUS],
                    [3., 0., 6., RADIUS],
                    [4., 0., 6., RADIUS],
                    [5., 0., 6., RADIUS],
                    [6., 0., 6., RADIUS]])

REF_LEN = 12


ROOT_NODE = Section(POINTS0)
ROOT_NODE.add_child(Section(POINTS1))


def test_init():
    nrt = Neurite(ROOT_NODE)
    nt.eq_(nrt.type, nm.NeuriteType.undefined)
    nt.eq_(len(nrt.points), 13)


def test_neurite_type():
    root_node = Section(POINTS0, section_type=nm.AXON)
    nrt = Neurite(root_node)
    nt.eq_(nrt.type, nm.AXON)

    root_node = Section(POINTS0, section_type=nm.BASAL_DENDRITE)
    nrt = Neurite(root_node)
    nt.eq_(nrt.type, nm.BASAL_DENDRITE)

    # https://github.com/BlueBrain/NeuroM/issues/697
    nt.assert_equal(np.array([nm.AXON, nm.BASAL_DENDRITE]).dtype,
                    np.object)


def test_neurite_length():
    nrt = Neurite(ROOT_NODE)
    nt.assert_almost_equal(nrt.length, REF_LEN)


def test_neurite_area():
    nrt = Neurite(ROOT_NODE)
    area = 2 * math.pi * RADIUS * REF_LEN
    nt.assert_almost_equal(nrt.area, area)


def test_neurite_volume():
    nrt = Neurite(ROOT_NODE)
    volume = math.pi * RADIUS * RADIUS * REF_LEN
    nt.assert_almost_equal(nrt.volume, volume)


def test_str():
    nrt = Neurite(ROOT_NODE)
    nt.ok_('Neurite' in str(nrt))


def test_neurite_hash():
    nrt = Neurite(ROOT_NODE)
    nt.eq_(hash(nrt), hash((nrt.type, nrt.root_node)))
