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

'''Test neurom.ezy.Neuron and neurom.features compatibility'''

import os
import numpy as np
from nose import tools as nt
from neurom import ezy
from neurom.ezy import NeuriteType
from neurom.features import get_feature as get



_PWD = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_PWD, '../../test_data/swc')
NRN = ezy.load_neuron(os.path.join(_DATA_PATH, 'Neuron.swc'))


def _equal(a, b):
    nt.assert_true(np.alltrue(a == b))


def test_get_section_lengths():
    _equal(NRN.get_section_lengths(), get('section_lengths', NRN))
    for t in NeuriteType:
        _equal(NRN.get_section_lengths(neurite_type=t), get('section_lengths', NRN, neurite_type=t))


def test_get_segment_lengths():
    _equal(NRN.get_segment_lengths(), get('segment_lengths', NRN))
    for t in NeuriteType:
        _equal(NRN.get_segment_lengths(neurite_type=t), get('segment_lengths', NRN, neurite_type=t))


def test_get_soma_radius():
    _equal(NRN.get_soma_radius(), get('soma_radii', NRN))


def test_get_soma_surface_area():
    _equal(NRN.get_soma_surface_area(), get('soma_surface_areas', NRN))


def test_get_local_bifurcation_angles():
    _equal(NRN.get_local_bifurcation_angles(), get('local_bifurcation_angles', NRN))
    for t in NeuriteType:
        _equal(NRN.get_local_bifurcation_angles(neurite_type=t), get('local_bifurcation_angles', NRN, neurite_type=t))


def test_get_remote_bifurcation_angles():
    _equal(NRN.get_remote_bifurcation_angles(), get('remote_bifurcation_angles', NRN))
    for t in NeuriteType:
        _equal(NRN.get_remote_bifurcation_angles(neurite_type=t), get('remote_bifurcation_angles', NRN, neurite_type=t))


def test_get_section_radial_distances():
    _equal(NRN.get_section_radial_distances(), get('section_radial_distances', NRN))
    for t in NeuriteType:
        _equal(NRN.get_section_radial_distances(neurite_type=t), get('section_radial_distances', NRN, neurite_type=t))


def test_get_section_path_distances():
    _equal(NRN.get_section_path_distances(), get('section_path_distances', NRN))
    for t in NeuriteType:
        _equal(NRN.get_section_path_distances(neurite_type=t), get('section_path_distances', NRN, neurite_type=t))


def test_get_n_sections():
    _equal(NRN.get_n_sections(), get('number_of_sections', NRN))
    for t in NeuriteType:
        _equal(NRN.get_n_sections(neurite_type=t), get('number_of_sections', NRN, neurite_type=t))


def test_get_n_sections_per_neurite():
    _equal(NRN.get_n_sections_per_neurite(), get('number_of_sections_per_neurite', NRN))
    for t in NeuriteType:
        _equal(NRN.get_n_sections_per_neurite(neurite_type=t), get('number_of_sections_per_neurite', NRN, neurite_type=t))


def test_get_number_of_neurites():
    _equal(NRN.get_n_neurites(), get('number_of_neurites', NRN))
    for t in NeuriteType:
        _equal(NRN.get_n_neurites(neurite_type=t), get('number_of_neurites', NRN, neurite_type=t))


def test_get_trunk_origin_radii():
    _equal(NRN.get_trunk_origin_radii(), get('trunk_origin_radii', NRN))
    for t in NeuriteType:
        _equal(NRN.get_trunk_origin_radii(neurite_type=t), get('trunk_origin_radii', NRN, neurite_type=t))


def test_get_trunk_section_lengths():
    _equal(NRN.get_trunk_section_lengths(), get('trunk_section_lengths', NRN))
    for t in NeuriteType:
        _equal(NRN.get_trunk_section_lengths(neurite_type=t), get('trunk_section_lengths', NRN, neurite_type=t))
