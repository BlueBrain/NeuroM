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
from neurom.io import ROOT_ID, COLS
from neurom.point_neurite.io import swc
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')


def check_single_section_random_swc(data, fmt):
    nt.ok_(fmt == 'SWC')
    nt.ok_(len(data) == 16)
    nt.ok_(np.shape(data) == (16, 7))


def test_read_swc_basic():
    rdw = swc.read(
        os.path.join(SWC_PATH,
                     'random_trunk_off_0_16pt.swc'))

    check_single_section_random_swc(rdw.data_block, rdw.fmt)


class TestRawDataWrapper_SingleSectionRandom(object):
    def setup(self):
        self.data = swc.read(
            os.path.join(SWC_PATH, 'sequential_trunk_off_42_16pt.swc'))
        self.first_id = int(self.data.data_block[0][COLS.ID])

    def test_data_structure(self):
        check_single_section_random_swc(self.data.data_block,
                                        self.data.fmt)

    def test_get_ids(self):
        nt.ok_(self.data.get_ids() == range(self.first_id, self.first_id+16))

    def test_get_ids_with_pred(self):
        nt.assert_equal(self.data.get_ids(lambda r: r[COLS.TYPE] == 2),
                        [self.first_id+2, self.first_id+10])

    @nt.raises(LookupError)
    def test_get_parent_invalid_id_raises(self):
        self.data.get_parent(-1)
        self.data.get_parent(-2)
        self.data.get_parent(16)

    @nt.raises(LookupError)
    def test_get_children_invalid_id_raises(self):
        self.data.get_children(-2)

    def test_fork_points_is_empty(self):
        nt.ok_(len(self.data.get_fork_points()) == 0)

    def test_get_parent(self):
        for i, idx in enumerate(self.data.get_ids()):
            if i == 0:
                nt.assert_equal(self.data.get_parent(idx), -1)
            else:
                nt.assert_equal(self.data.get_parent(idx), idx - 1)

    def test_get_children(self):
        ids = self.data.get_ids()
        last = ids[-1]
        for i in self.data.get_ids():
            children = self.data.get_children(i)
            if i != last:
                nt.ok_(children == [i + 1])
            else:
                nt.ok_(len(children) == 0)

        nt.ok_(self.data.get_children(ROOT_ID) == [ids[0]])

    def test_get_endpoints(self):
        # end-point is last point
        nt.ok_(self.data.get_end_points() == [self.data.data_block[-1][COLS.ID]])


    def test_get_row(self):
        for i in self.data.get_ids():
            r = self.data.get_row(i)
            ii = i - self.first_id
            nt.ok_(len(r) == 7)
            nt.ok_(r[4] >= 0 and r[4] < 8)
            nt.ok_(r[4] == ii % 8)
            nt.ok_(r[0] == ii)
            nt.ok_(r[1] == ii)
            nt.ok_(r[2] == ii)
            nt.ok_(r[3] == ii)

    def test_get_point(self):
        for i in self.data.get_ids():
            p = self.data.get_point(i)
            ii = i - self.first_id
            nt.ok_(p)
            nt.ok_(p.t >= 0 and p.t < 8)
            nt.ok_(p.t == ii % 8)
            nt.ok_(p.x == ii)
            nt.ok_(p.y == ii)
            nt.ok_(p.z == ii)
            nt.ok_(p.r == ii)

    def test_iter_row(self):
        for i, p in enumerate(self.data.iter_row()):
            ii = i + self.first_id
            pid = -1 if i == 0 else ii - 1
            nt.assert_true(np.all(p == (i, i, i, i, i%8, ii, pid)))

        for p in self.data.iter_row(None, lambda r: r[COLS.TYPE] == 1):
            nt.assert_true(p[COLS.TYPE] == 1)

    @nt.raises(LookupError)
    def test_iter_row_low_id_raises(self):
        self.data.iter_row(-1)

    @nt.raises(LookupError)
    def test_iter_row_high_id_raises(self):
        self.data.iter_row(16 + self.first_id)
