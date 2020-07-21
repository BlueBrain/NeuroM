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

"""Test neurom.sectionfunc functionality."""

from nose import tools as nt
from pathlib import Path
import math
import numpy as np
import warnings
from io import StringIO
from numpy.testing import assert_allclose
from neurom import load_neuron

# NOTE: The 'bf' alias is used in the fst/tests modules
# Do NOT change it.
# TODO: If other neurom.features are imported,
# the should use the aliasing used in fst/tests module files
from neurom.features import sectionfunc as _sf
from neurom.features import neuritefunc as _nf
from neurom import morphmath as mmth

DATA_PATH = Path(__file__).parent.parent.parent.parent / 'test_data'
H5_PATH = Path(DATA_PATH, 'h5/v1/')
SWC_PATH = Path(DATA_PATH, 'swc/')

NRN = load_neuron(H5_PATH / 'Neuron.h5')


def test_total_volume_per_neurite():

    vol = _nf.total_volume_per_neurite(NRN)
    nt.eq_(len(vol), 4)

    # calculate the volumes by hand and compare
    vol2 = [sum(_sf.section_volume(s) for s in n.iter_sections()) for n in NRN.neurites]
    nt.eq_(vol, vol2)

    # regression test
    ref_vol = [271.94122143951864, 281.24754646913954,
               274.98039928781355, 276.73860261723024]
    assert_allclose(vol, ref_vol)


def test_section_area():
    sec = load_neuron(StringIO(u"""((CellBody) (0 0 0 2))
                                    ((Dendrite)
                                     (0 0 0 2)
                                     (1 0 0 2))"""), reader='asc').sections[1]
    area = _sf.section_area(sec)
    nt.eq_(math.pi * 1 * 2 * 1, area)


def test_section_tortuosity():
    sec_a = load_neuron(StringIO(u"""
	((CellBody) (0 0 0 2))
	((Dendrite)
    (0 0 0 2)
    (1 0 0 2)
    (2 0 0 2)
    (3 0 0 2))"""), reader='asc').sections[1]

    sec_b = load_neuron(StringIO(u"""
    ((CellBody) (0 0 0 2))
    ((Dendrite)
    (0 0 0 2)
    (1 0 0 2)
    (1 2 0 2)
    (0 2 0 2))"""), reader='asc').sections[1]

    nt.eq_(_sf.section_tortuosity(sec_a), 1.0)
    nt.eq_(_sf.section_tortuosity(sec_b), 4.0 / 2.0)

    for s in _nf.iter_sections(NRN):
        nt.eq_(_sf.section_tortuosity(s),
               mmth.section_length(s.points) / mmth.point_dist(s.points[0],
                                                               s.points[-1]))

def test_setion_tortuosity_single_point():
    sec = load_neuron(StringIO(u"""((CellBody) (0 0 0 2))
                                   ((Dendrite)
                                    (1 2 3 2))"""), reader='asc').sections[1]
    nt.eq_(_sf.section_tortuosity(sec), 1.0)


def test_section_tortuosity_looping_section():
    sec = load_neuron(StringIO(u"""
    ((CellBody) (0 0 0 2))
    ((Dendrite)
    (0 0 0 2)
    (1 0 0 2)
    (1 2 0 2)
    (0 2 0 2)
    (0 0 0 2))"""), reader='asc').sections[1]
    with warnings.catch_warnings(record=True):
        nt.eq_(_sf.section_tortuosity(sec), np.inf)


def test_section_meander_angles():
    s0 = load_neuron(StringIO(u"""((CellBody) (0 0 0 0))
    ((Dendrite)
    (0 0 0 2)
    (1 0 0 2)
    (2 0 0 2)
    (3 0 0 2)
    (4 0 0 2))"""), reader='asc').sections[1]

    nt.assert_equal(_sf.section_meander_angles(s0),
                    [math.pi, math.pi, math.pi])

    s1 = load_neuron(StringIO(u"""((CellBody) (0 0 0 0))
    ((Dendrite)
    (0 0 0 2)
    (1 0 0 2)
    (1 1 0 2)
    (2 1 0 2)
    (2 2 0 2))"""), reader='asc').sections[1]

    nt.assert_equal(_sf.section_meander_angles(s1),
                    [math.pi / 2, math.pi / 2, math.pi / 2])

    s2 = load_neuron(StringIO(u"""((CellBody) (0 0 0 0))
    ((Dendrite)
    (0 0 0 2)
    (0 0 1 2)
    (0 0 2 2)
    (0 0 0 2))"""), reader='asc').sections[1]

    nt.assert_equal(_sf.section_meander_angles(s2),
                    [math.pi, 0.])


def test_section_meander_angles_single_segment():
    s = load_neuron(StringIO(u"""((CellBody) (0 0 0 0))
    ((Dendrite)
    (0 0 0 2)
    (1 1 1 2))"""), reader='asc').sections[1]
    nt.assert_equal(len(_sf.section_meander_angles(s)), 0)


def test_strahler_order():
    path = Path(SWC_PATH, 'strahler.swc')
    n = load_neuron(path)
    strahler_order = _sf.strahler_order(n.neurites[0].root_node)
    nt.eq_(strahler_order, 4)


def test_locate_segment_position():
    s = load_neuron(StringIO(u"""((CellBody) (0 0 0 0))
    ((Dendrite)
    (0 0 0 0)
    (3 0 4 200)
    (6 4 4 400))"""), reader='asc').sections[1]
    nt.assert_equal(
        _sf.locate_segment_position(s, 0.0),
        (0, 0.0)
    )
    nt.assert_equal(
        _sf.locate_segment_position(s, 0.25),
        (0, 2.5)
    )
    nt.assert_equal(
        _sf.locate_segment_position(s, 0.75),
        (1, 2.5)
    )
    nt.assert_equal(
        _sf.locate_segment_position(s, 1.0),
        (1, 5.0)
    )
    nt.assert_raises(
        ValueError,
        _sf.locate_segment_position, s, 1.1
    )
    nt.assert_raises(
        ValueError,
        _sf.locate_segment_position, s, -0.1
    )

def test_mean_radius():
    s = load_neuron(StringIO(u"""
    ((CellBody)
     (0 0 0 1))

    ((Dendrite)
    (0 0 0 0)
    (3 0 4 200)
    (6 4 4 400))"""), reader='asc').neurites[0]

    nt.assert_equal(
        _sf.section_mean_radius(s),
       100.
    )
