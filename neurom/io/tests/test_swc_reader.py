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

from pathlib import Path

import numpy as np

from neurom.core.dataformat import COLS
from neurom.io import swc
from neurom import load_neuron, NeuriteType

from nose import tools as nt
from nose.tools import assert_equal


DATA_PATH = Path(__file__).parent.parent.parent.parent / 'test_data'
SWC_PATH = Path(DATA_PATH, 'swc')
SWC_SOMA_PATH = Path(SWC_PATH, 'soma')


def test_read_swc_basic_with_offset_0():
    """first ID = 0, rare to find."""
    rdw = swc.read(Path(SWC_PATH, 'sequential_trunk_off_0_16pt.swc'))
    nt.eq_(rdw.fmt, 'SWC')
    nt.eq_(len(rdw.data_block), 16)
    nt.eq_(np.shape(rdw.data_block), (16, 7))


def test_read_swc_basic_with_offset_1():
    """More normal ID numbering, starting at 1."""
    rdw = swc.read(Path(SWC_PATH, 'sequential_trunk_off_1_16pt.swc'))
    nt.eq_(rdw.fmt, 'SWC')
    nt.eq_(len(rdw.data_block), 16)
    nt.eq_(np.shape(rdw.data_block), (16, 7))


def test_read_swc_basic_with_offset_42():
    """ID numbering starting at 42."""
    rdw = swc.read(Path(SWC_PATH, 'sequential_trunk_off_42_16pt.swc'))
    nt.eq_(rdw.fmt, 'SWC')
    nt.eq_(len(rdw.data_block), 16)
    nt.eq_(np.shape(rdw.data_block), (16, 7))


def test_read_single_neurite():
    rdw = swc.read(Path(SWC_PATH, 'point_soma_single_neurite.swc'))
    nt.eq_(rdw.neurite_root_section_ids(), [1])
    nt.eq_(len(rdw.soma_points()), 1)
    nt.eq_(len(rdw.sections), 2)


def test_read_split_soma():
    rdw = swc.read(Path(SWC_PATH, 'split_soma_single_neurites.swc'))
    nt.eq_(rdw.neurite_root_section_ids(), [1, 3])
    nt.eq_(len(rdw.soma_points()), 3)
    nt.eq_(len(rdw.sections), 4)

    ref_ids = [[-1, 0],
               [0, 1, 2, 3, 4],
               [0, 5, 6],
               [6, 7, 8, 9, 10],
               []]

    for s, r in zip(rdw.sections, ref_ids):
        nt.eq_(s.ids, r)


def test_simple_reversed():
    rdw = swc.read(Path(SWC_PATH, 'simple_reversed.swc'))
    nt.eq_(rdw.neurite_root_section_ids(), [5, 6])
    nt.eq_(len(rdw.soma_points()), 1)
    nt.eq_(len(rdw.sections), 7)

def test_custom_type():
    neuron = load_neuron(Path(SWC_PATH, 'custom_type.swc'))
    assert_equal(neuron.neurites[1].type, NeuriteType.custom)

def test_undefined_type():
    neuron = load_neuron(Path(SWC_PATH, 'undefined_type.swc'))
    assert_equal(neuron.neurites[1].type, NeuriteType.undefined)
