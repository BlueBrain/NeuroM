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

'''Test neurom.features and section tree features compatibility'''

import os
import numpy as np
from nose import tools as nt
from neurom.core.types import NeuriteType
from neurom.core import section_neuron as sn
from neurom.core.tree import i_chain2, ibifurcation_point
from neurom.io.utils import load_neuron
from neurom.features import get
from neurom.analysis import morphtree as mt



_PWD = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_PWD, '../../../test_data/h5/v1/Neuron.h5')

SEC_NRN = sn.load_neuron(_DATA_PATH)
REF_NRN = load_neuron(_DATA_PATH, mt.set_tree_type)

REF_NEURITE_TYPES = [NeuriteType.apical_dendrite, NeuriteType.basal_dendrite,
                     NeuriteType.basal_dendrite, NeuriteType.axon]

def _close(a, b, debug=False):
    if debug:
        print 'a: %s\nb:%s\n' % (a, b)
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.allclose(a, b))


def _equal(a, b, debug=False):
    if debug:
        print 'a: %s\nb:%s\n' % (a, b)
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.alltrue(a == b))


def test_neurite_type():

    neurite_types = [n0.type for n0 in SEC_NRN.neurites]
    nt.assert_equal(neurite_types, REF_NEURITE_TYPES)
    nt.assert_equal(neurite_types, [n1.type for n1 in REF_NRN.neurites])


def test_get_n_sections():
    nt.assert_equal(sn.n_sections(SEC_NRN), get('number_of_sections', REF_NRN)[0])
    for t in NeuriteType:
        nt.assert_equal(sn.n_sections(SEC_NRN, neurite_type=t),
                        get('number_of_sections', REF_NRN, neurite_type=t)[0])


def test_get_n_sections_per_neurite():
    _equal(sn.get_n_sections_per_neurite(SEC_NRN),
           get('number_of_sections_per_neurite', REF_NRN))

    for t in NeuriteType:
        _equal(sn.get_n_sections_per_neurite(SEC_NRN, neurite_type=t),
               get('number_of_sections_per_neurite', REF_NRN, neurite_type=t))


def test_get_n_segments():
    nt.assert_equal(sn.n_segments(SEC_NRN), get('number_of_segments', REF_NRN)[0])
    for t in NeuriteType:
        nt.assert_equal(sn.n_segments(SEC_NRN, neurite_type=t),
                        get('number_of_segments', REF_NRN, neurite_type=t)[0])


def test_get_number_of_neurites():
    nt.assert_equal(sn.n_neurites(SEC_NRN), get('number_of_neurites', REF_NRN)[0])
    for t in NeuriteType:
        nt.assert_equal(sn.n_neurites(SEC_NRN, neurite_type=t),
                        get('number_of_neurites', REF_NRN, neurite_type=t)[0])


def test_get_section_lengths():
    _close(sn.get_section_lengths(SEC_NRN), get('section_lengths', REF_NRN))
    for t in NeuriteType:
        _close(sn.get_section_lengths(SEC_NRN, neurite_type=t),
               get('section_lengths', REF_NRN, neurite_type=t))


def test_get_section_path_distances():
    _close(sn.get_path_lengths(SEC_NRN), get('section_path_distances', REF_NRN))
    for t in NeuriteType:
        _close(sn.get_path_lengths(SEC_NRN, neurite_type=t),
               get('section_path_distances', REF_NRN, neurite_type=t))

    pl = [sn.section_path_length(s) for s in i_chain2(SEC_NRN.neurites)]
    _close(pl, get('section_path_distances', REF_NRN))


@nt.nottest
def test_get_segment_lengths():
    _equal(sn.get_segment_lengths(SEC_NRN), get('segment_lengths', REF_NRN))
    for t in NeuriteType:
        _equal(sn.get_segment_lengths(SEC_NRN, neurite_type=t),
               get('segment_lengths', REF_NRN, neurite_type=t))


def test_get_soma_radius():
    nt.assert_equal(SEC_NRN.soma.radius, get('soma_radii', REF_NRN)[0])


def test_get_soma_surface_area():
    nt.assert_equal(sn.soma_surface_area(SEC_NRN), get('soma_surface_areas', REF_NRN)[0])


def test_get_local_bifurcation_angles():
    _close(sn.get_local_bifurcation_angles(SEC_NRN),
           get('local_bifurcation_angles', REF_NRN))

    for t in NeuriteType:
        _close(sn.get_local_bifurcation_angles(SEC_NRN, neurite_type=t),
               get('local_bifurcation_angles', REF_NRN, neurite_type=t))

    ba = [sn.local_bifurcation_angle(b)
          for b in i_chain2(SEC_NRN.neurites, iterator_type=ibifurcation_point)]

    _close(ba, get('local_bifurcation_angles', REF_NRN))


def test_get_remote_bifurcation_angles():
    _close(sn.get_remote_bifurcation_angles(SEC_NRN),
           get('remote_bifurcation_angles', REF_NRN))

    for t in NeuriteType:
        _close(sn.get_remote_bifurcation_angles(SEC_NRN, neurite_type=t),
               get('remote_bifurcation_angles', REF_NRN, neurite_type=t))

    ba = [sn.remote_bifurcation_angle(b)
          for b in i_chain2(SEC_NRN.neurites, iterator_type=ibifurcation_point)]

    _close(ba, get('remote_bifurcation_angles', REF_NRN))


def test_get_section_radial_distances():
    _close(sn.get_section_radial_distances(SEC_NRN),
           get('section_radial_distances', REF_NRN))

    for t in NeuriteType:
        _close(sn.get_section_radial_distances(SEC_NRN, neurite_type=t),
               get('section_radial_distances', REF_NRN, neurite_type=t))


def test_get_trunk_origin_radii():
    _equal(sn.get_trunk_origin_radii(SEC_NRN), get('trunk_origin_radii', REF_NRN))
    for t in NeuriteType:
        _equal(sn.get_trunk_origin_radii(SEC_NRN, neurite_type=t),
               get('trunk_origin_radii', REF_NRN, neurite_type=t))


def test_get_trunk_section_lengths():
    _equal(sn.get_trunk_section_lengths(SEC_NRN), get('trunk_section_lengths', REF_NRN))
    for t in NeuriteType:
        _equal(sn.get_trunk_section_lengths(SEC_NRN, neurite_type=t),
               get('trunk_section_lengths', REF_NRN, neurite_type=t))
