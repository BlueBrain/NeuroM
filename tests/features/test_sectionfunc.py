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

"""Test neurom.sectionfunc."""

import math
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
from neurom import load_morphology, iter_sections
from neurom import morphmath
from neurom.features import section

import pytest

DATA_PATH = Path(__file__).parent.parent / 'data'
H5_PATH = DATA_PATH / 'h5/v1/'
SWC_PATH = DATA_PATH / 'swc/'
NRN = load_morphology(H5_PATH / 'Neuron.h5')
SECTION_ID = 0


def test_section_area():
    sec = load_morphology(StringIO(u"""((CellBody) (0 0 0 2))
                                    ((Dendrite)
                                     (0 0 0 2)
                                     (1 0 0 2))"""), reader='asc').sections[SECTION_ID]
    area = section.section_area(sec)
    assert math.pi * 1 * 2 * 1 == area


def test_section_tortuosity():
    sec_a = load_morphology(StringIO(u"""
	((CellBody) (0 0 0 2))
	((Dendrite)
    (0 0 0 2)
    (1 0 0 2)
    (2 0 0 2)
    (3 0 0 2))"""), reader='asc').sections[SECTION_ID]

    sec_b = load_morphology(StringIO(u"""
    ((CellBody) (0 0 0 2))
    ((Dendrite)
    (0 0 0 2)
    (1 0 0 2)
    (1 2 0 2)
    (0 2 0 2))"""), reader='asc').sections[SECTION_ID]

    assert section.section_tortuosity(sec_a) == 1.0
    assert section.section_tortuosity(sec_b) == 4.0 / 2.0

    for s in iter_sections(NRN):
        assert (section.section_tortuosity(s) ==
                morphmath.section_length(s.points) / morphmath.point_dist(s.points[0], s.points[-1]))

def test_setion_tortuosity_single_point():
    sec = load_morphology(StringIO(u"""((CellBody) (0 0 0 2))
                                   ((Dendrite)
                                    (1 2 3 2))"""), reader='asc').sections[SECTION_ID]
    assert section.section_tortuosity(sec) == 1.0


def test_section_tortuosity_looping_section():
    sec = load_morphology(StringIO(u"""
    ((CellBody) (0 0 0 2))
    ((Dendrite)
    (0 0 0 2)
    (1 0 0 2)
    (1 2 0 2)
    (0 2 0 2)
    (0 0 0 2))"""), reader='asc').sections[SECTION_ID]
    with warnings.catch_warnings(record=True):
        assert section.section_tortuosity(sec) == np.inf


def test_section_meander_angles():
    s0 = load_morphology(StringIO(u"""((CellBody) (0 0 0 0))
    ((Dendrite)
    (0 0 0 2)
    (1 0 0 2)
    (2 0 0 2)
    (3 0 0 2)
    (4 0 0 2))"""), reader='asc').sections[SECTION_ID]
    assert section.section_meander_angles(s0) == [math.pi, math.pi, math.pi]

    s1 = load_morphology(StringIO(u"""((CellBody) (0 0 0 0))
    ((Dendrite)
    (0 0 0 2)
    (1 0 0 2)
    (1 1 0 2)
    (2 1 0 2)
    (2 2 0 2))"""), reader='asc').sections[SECTION_ID]
    assert section.section_meander_angles(s1) == [math.pi / 2, math.pi / 2, math.pi / 2]

    s2 = load_morphology(StringIO(u"""((CellBody) (0 0 0 0))
    ((Dendrite)
    (0 0 0 2)
    (0 0 1 2)
    (0 0 2 2)
    (0 0 0 2))"""), reader='asc').sections[SECTION_ID]
    assert section.section_meander_angles(s2) == [math.pi, 0.]


def test_section_meander_angles_single_segment():
    s = load_morphology(StringIO(u"""((CellBody) (0 0 0 0))
    ((Dendrite)
    (0 0 0 2)
    (1 1 1 2))"""), reader='asc').sections[SECTION_ID]
    assert len(section.section_meander_angles(s)) == 0


def test_strahler_order():
    path = Path(SWC_PATH, 'strahler.swc')
    n = load_morphology(path)
    strahler_order = section.strahler_order(n.neurites[0].root_node)
    assert strahler_order == 4


def test_locate_segment_position():
    s = load_morphology(StringIO(u"""((CellBody) (0 0 0 0))
    ((Dendrite)
    (0 0 0 0)
    (3 0 4 200)
    (6 4 4 400))"""), reader='asc').sections[SECTION_ID]

    assert section.locate_segment_position(s, 0.0) == (0, 0.0)
    assert section.locate_segment_position(s, 0.25) == (0, 2.5)
    assert section.locate_segment_position(s, 0.75) == (1, 2.5)
    assert section.locate_segment_position(s, 1.0) == (1, 5.0)

    with pytest.raises(ValueError):
        section.locate_segment_position(s, 1.1)
    with pytest.raises(ValueError):
        section.locate_segment_position(s, -0.1)


def test_mean_radius():
    n = load_morphology(StringIO(u"""
    ((CellBody)
     (0 0 0 1))

    ((Dendrite)
    (0 0 0 0)
    (3 0 4 200)
    (6 4 4 400))"""), reader='asc')

    assert section.section_mean_radius(n.neurites[0]) == 100.
