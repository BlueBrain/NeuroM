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
from neurom.io import COLS
from neurom.io import swc
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')
SWC_SOMA_PATH = os.path.join(SWC_PATH, 'soma')


def check_single_section_random_swc(data, fmt):
    nt.ok_(fmt == 'SWC')
    nt.ok_(len(data) == 16)
    nt.ok_(np.shape(data) == (16, 7))


def test_read_swc_basic():
    rdw = swc.read(
        os.path.join(SWC_PATH,
                     'random_trunk_off_0_16pt.swc'))

    check_single_section_random_swc(rdw.data_block, rdw.fmt)


def test_read_single_neurite():
    rdw = swc.read(os.path.join(SWC_PATH, 'point_soma_single_neurite.swc'))
    nt.eq_(rdw.neurite_root_section_ids(), [1])
    nt.eq_(len(rdw.soma_points()), 1)
    nt.eq_(len(rdw.sections), 3) # includes one empty section


def test_read_split_soma():
    rdw = swc.read(os.path.join(SWC_PATH, 'split_soma_single_neurites.swc'))
    nt.eq_(rdw.neurite_root_section_ids(), [1, 3])
    nt.eq_(len(rdw.soma_points()), 3)
    nt.eq_(len(rdw.sections), 5) # includes one empty section

    ref_ids = [[-1, 0],
               [0, 1, 2, 3, 4],
               [0, 5, 6],
               [6, 7, 8, 9, 10],
               []]

    for s, r in zip(rdw.sections, ref_ids):
        nt.eq_(s.ids, r)


def test_read_contour_soma_neuron():
    rdw = swc.read(os.path.join(SWC_SOMA_PATH, 'contour_soma_neuron.swc'))
    nt.eq_(rdw.neurite_root_section_ids(), [3, 4, 5])
    nt.eq_(len(rdw.soma_points()), 8)
    nt.eq_(len(rdw.sections), 7) # includes one empty section

    ref_ids = [[-1, 0, 1, 2],
               [2, 3, 4, 5],
               [5, 6, 7],
               [2, 8, 9, 10, 11, 12],
               [5, 13, 14, 15, 16, 17],
               [7, 18, 19, 20, 21, 22],
               []]

    for s, r in zip(rdw.sections, ref_ids):
        nt.eq_(s.ids, r)


def test_read_contour_split_soma_neuron():
    rdw = swc.read(os.path.join(SWC_SOMA_PATH, 'contour_split_soma_neuron.swc'))
    nt.eq_(rdw.neurite_root_section_ids(), [1, 4, 5])
    nt.eq_(len(rdw.soma_points()), 8)
    nt.eq_(len(rdw.sections), 7) # includes one empty section

    ref_ids = [[-1, 0, 1, 2],
               [2, 3, 4, 5, 6, 7],
               [2, 8, 9, 10],
               [10, 11, 12],
               [10, 13, 14, 15, 16, 17],
               [12, 18, 19, 20, 21, 22],
               []]

    for s, r in zip(rdw.sections, ref_ids):
        nt.eq_(s.ids, r)


def test_read_contour_split_1st_soma_neuron():
    rdw = swc.read(os.path.join(SWC_SOMA_PATH, 'contour_split_1st_soma_neuron.swc'))
    nt.eq_(rdw.neurite_root_section_ids(), [1, 4, 5])
    nt.eq_(len(rdw.soma_points()), 6)
    nt.eq_(len(rdw.sections), 7) # includes one empty section

    ref_ids = [[-1, 0],
               [0, 1, 2, 3, 4, 5, 6, 7],
               [0, 8, 9, 10],
               [10, 11, 12],
               [10, 13, 14, 15, 16, 17],
               [12, 18, 19, 20, 21, 22],
               []]

    for s, r in zip(rdw.sections, ref_ids):
        nt.eq_(s.ids, r)


class TestDataWrapper_SingleSectionRandom(object):
    def setup(self):
        self.data = swc.read(
            os.path.join(SWC_PATH, 'sequential_trunk_off_42_16pt.swc'))
        self.first_id = int(self.data.data_block[0][COLS.ID])

    def test_data_structure(self):
        check_single_section_random_swc(self.data.data_block,
                                        self.data.fmt)
