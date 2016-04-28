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
from neurom.core.tree import i_chain2
from neurom.io.utils import load_neuron
from neurom.features import get



_PWD = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_PWD, '../../../test_data/h5/v1/Neuron.h5')

SEC_NRN = sn.load_neuron(_DATA_PATH)
REF_NRN = load_neuron(_DATA_PATH)


def _close(a, b):
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.allclose(a, b))


def _equal(a, b):
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.alltrue(a == b))


def test_get_n_sections():
    nt.assert_equal(sn.n_sections(SEC_NRN), get('number_of_sections', REF_NRN)[0])


def test_get_n_segments():
    nt.assert_equal(sn.n_segments(SEC_NRN), get('number_of_segments', REF_NRN)[0])


def test_get_number_of_neurites():
    nt.assert_equal(sn.n_neurites(SEC_NRN), get('number_of_neurites', REF_NRN)[0])


def test_get_section_lengths():
    _close(sn.get_section_lengths(SEC_NRN), get('section_lengths', REF_NRN))


def test_get_section_path_distances():
    _close(sn.get_path_lengths(SEC_NRN), get('section_path_distances', REF_NRN))

    pl = [sn.path_length(s) for s in i_chain2(SEC_NRN.neurites)]
    _close(pl, get('section_path_distances', REF_NRN))


@nt.nottest
def test_get_segment_lengths():
    nt.assert_true(False)


@nt.nottest
def test_get_soma_radius():
    nt.assert_true(False)


@nt.nottest
def test_get_soma_surface_area():
    nt.assert_true(False)


@nt.nottest
def test_get_local_bifurcation_angles():
    nt.assert_true(False)


@nt.nottest
def test_get_remote_bifurcation_angles():
    nt.assert_true(False)


@nt.nottest
def test_get_section_radial_distances():
    nt.assert_true(False)
