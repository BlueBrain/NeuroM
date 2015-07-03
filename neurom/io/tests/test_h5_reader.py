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
from neurom.io import readers
from neurom.core.dataformat import COLS
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
H5_PATH = os.path.join(DATA_PATH, 'h5')


def test_read_h5v1_basic():
    data, offset, fmt = readers.H5V1.read(
        os.path.join(H5_PATH, 'Neuron_v1.h5'))

    nt.ok_(fmt == 'H5V1')
    nt.ok_(offset == 0)
    nt.ok_(len(data) == 482)
    nt.ok_(np.shape(data) == (482, 7))


class TestRawDataWrapper_Neuron_H5V1(object):
    def setup(self):
        self.data = readers.load_data(
            os.path.join(H5_PATH, 'Neuron_v1.h5'))
        self.first_id = int(self.data.data_block[0][COLS.ID])
        self.rows = len(self.data.data_block)

    def test_n_rows(self):
        nt.ok_(self.rows == 482)

    def test_first_id_0(self):
        nt.ok_(self.first_id == 0)

    def test_fork_points(self):
        fpts = [20, 30, 52, 74, 96, 118, 140, 162, 184, 206,
                228, 261, 283, 305, 327, 349, 371, 393, 415, 437, 459]
        nt.assert_equal(len(self.data.get_fork_points()), 21)
        nt.assert_equal(self.data.get_fork_points(), fpts)

    @nt.raises(LookupError)
    def test_iter_row_low_id_raises(self):
        self.data.iter_row(-1)

    @nt.raises(LookupError)
    def test_iter_row_high_id_raises(self):
        self.data.iter_row(self.rows + self.first_id)
