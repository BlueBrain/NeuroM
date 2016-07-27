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
from nose import tools as nt
from neurom.point_neurite.io.utils import load_neuron
from neurom.core.types import NeuriteType
from neurom import fst
from neurom import _compat
from neurom.analysis.morphmath import segment_radius as segrad


_PWD = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_PWD, '../../test_data/h5/v1/Neuron.h5')

NRN0 = load_neuron(DATA_PATH)
NRN1 = fst.load_neuron(DATA_PATH)


def _close(a, b, debug=False):
    if debug:
        print '\na.shape: %s\nb.shape: %s\n' % (a.shape, b.shape)
        print '\na: %s\nb:%s\n' % (a, b)
        print '\na - b:%s\n' % (a - b)
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.allclose(a, b))


def _equal(a, b, debug=False):
    if debug:
        print '\na.shape: %s\nb.shape: %s\n' % (a.shape, b.shape)
        print '\na: %s\nb:%s\n' % (a, b)
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.alltrue(a == b))


def test_is_new_style_true():
    nt.assert_true(_compat.is_new_style(NRN1.neurites[0]))
    nt.assert_true(_compat.is_new_style(NRN1))


def test_is_new_style_false():
    nt.assert_false(_compat.is_new_style(NRN0.neurites[0]))
    nt.assert_false(_compat.is_new_style(NRN0))
    nt.assert_false(_compat.is_new_style([1,2,3]))


def test_bounding_box():

    ref_bbox = [[0.0, -52.79933167, 0.], [64.74726105, 0, 54.2040863]]
    bbox0 = _compat.bounding_box(NRN0.neurites[0])
    bbox1 = _compat.bounding_box(NRN1.neurites[0])
    bbox2 = _compat.bounding_box(NRN1.neurites[0].root_node)
    _close(bbox0, ref_bbox)
    _close(bbox1, ref_bbox)
    _close(bbox2, ref_bbox)
    _equal(bbox0, bbox1)
    _equal(bbox1, bbox2)


def test_map_segments():
    rad0 = _compat.map_segments(NRN0.neurites[0], segrad)
    rad1 = _compat.map_segments(NRN1.neurites[0], segrad)
    rad2 = _compat.map_segments(NRN1.neurites[0].root_node, segrad)
    nt.assert_equal(len(rad0), 210)
    _equal(rad0, rad1)
    _equal(rad1, rad2)


def test_neurite_type():

    ref_types = [NeuriteType.apical_dendrite, NeuriteType.basal_dendrite,
                 NeuriteType.basal_dendrite, NeuriteType.axon]

    ntypes0 = [_compat.neurite_type(n) for n in NRN0.neurites]
    ntypes1 = [_compat.neurite_type(n) for n in NRN1.neurites]

    _equal(ntypes0, ref_types)
    _equal(ntypes1, ref_types)
    _equal(ntypes0, ntypes1)
