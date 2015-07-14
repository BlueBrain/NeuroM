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
import h5py
from neurom.io import readers
from neurom.io import hdf5
from neurom.core.dataformat import COLS
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
H5_PATH = os.path.join(DATA_PATH, 'h5')
H5V1_PATH = os.path.join(H5_PATH, 'v1')
H5V2_PATH = os.path.join(H5_PATH, 'v2')


def test_read_h5v1_basic():
    data, offset, fmt = readers.H5.read(
        os.path.join(H5V1_PATH, 'Neuron.h5'))

    nt.ok_(fmt == 'H5V1')
    nt.ok_(offset == 0)
    nt.assert_equal(len(data), 927)
    nt.assert_equal(np.shape(data), (927, 7))


def test_read_h5v2_repaired_basic():
    data, offset, fmt = readers.H5.read(
        os.path.join(H5V2_PATH, 'Neuron_2_branch.h5'))

    nt.ok_(fmt == 'H5V2')
    nt.ok_(offset == 0)
    nt.assert_equal(len(data), 482)
    nt.assert_equal(np.shape(data), (482, 7))


def test_read_h5v2_raw_basic():
    data, offset, fmt = readers.H5.read(
        os.path.join(H5V2_PATH, 'Neuron.h5'))

    nt.ok_(fmt == 'H5V2')
    nt.ok_(offset == 0)
    nt.assert_equal(len(data), 927)
    nt.assert_equal(np.shape(data), (927, 7))


def test_get_version():
    v1 = h5py.File(os.path.join(H5V1_PATH, 'Neuron.h5'))
    v2 = h5py.File(os.path.join(H5V2_PATH, 'Neuron.h5'))
    nt.assert_equal(hdf5.get_version(v1), 'H5V1')
    nt.assert_equal(hdf5.get_version(v2), 'H5V2')
    v1.close()
    v2.close()


def test_unpack_h2():
    v1 = h5py.File(os.path.join(H5V1_PATH, 'Neuron.h5'))
    v2 = h5py.File(os.path.join(H5V2_PATH, 'Neuron.h5'))
    pts1, grp1 = hdf5._unpack_v1(v1)
    pts2, grp2 = hdf5._unpack_v2(v2, stage='raw')
    nt.assert_true(np.all(pts1 == pts2))
    nt.assert_true(np.all(grp1 == grp2))


class DataWrapper_Neuron(object):
    '''Base class for H5 tests'''

    end_pts = [1, 386, 133, 782, 529, 914, 276, 661, 23, 408, 155, 925,
               926, 804, 551, 298, 683, 45, 430, 177, 694, 826, 573, 320,
               67, 452, 199, 716, 463, 848, 595, 342, 89, 221, 738, 485,
               870, 232, 617, 364, 111, 760, 507, 892, 254, 639]

    end_parents = [0, 385, 132, 781, 528, 913, 275, 660, 22, 407, 154, 924,
                   0, 803, 550, 297, 682, 44, 429, 176, 693, 825, 572, 319,
                   66, 451, 198, 715, 462, 847, 594, 341, 88, 220, 737, 484,
                   869, 231, 616, 363, 110, 759, 506, 891, 253, 638]

    fork_pts = [0, 12, 34, 56, 78, 100, 122, 144, 166, 188, 210, 243, 265,
                287, 309, 331, 353, 375, 397, 419, 441, 474, 496, 518, 540,
                562, 584, 606, 628, 650, 672, 705, 727, 749, 771, 793, 815,
                837, 859, 881, 903]

    fork_parents = [-1, 11, 33, 55, 77, 99, 121, 143, 165, 187, 209, 242,
                    264, 286, 308, 330, 352, 374, 396, 418, 440, 473, 495,
                    517, 539, 561, 583, 605, 627, 649, 671, 704, 726, 748,
                    770, 792, 814, 836, 858, 880, 902]

    def test_n_rows(self):
        nt.assert_equal(self.rows, 927)

    def test_first_id_0(self):
        nt.ok_(self.first_id == 0)

    def test_fork_points(self):
        nt.assert_equal(len(self.data.get_fork_points()), 41)
        nt.assert_equal(self.data.get_fork_points(),
                        DataWrapper_Neuron.fork_pts)

    def test_get_endpoints(self):
        nt.assert_equal(self.data.get_end_points(),
                        DataWrapper_Neuron.end_pts)

    def test_end_points_have_no_children(self):
        for p in DataWrapper_Neuron.end_pts:
            nt.ok_(len(self.data.get_children(p)) == 0)

    def test_fork_point_parents(self):
        fpar = [self.data.get_parent(i) for i in self.data.get_fork_points()]
        nt.assert_equal(fpar, DataWrapper_Neuron.fork_parents)

    def test_end_point_parents(self):
        epar = [self.data.get_parent(i) for i in self.data.get_end_points()]
        nt.assert_equal(epar, DataWrapper_Neuron.end_parents)

    @nt.raises(LookupError)
    def test_iter_row_low_id_raises(self):
        self.data.iter_row(-1)

    @nt.raises(LookupError)
    def test_iter_row_high_id_raises(self):
        self.data.iter_row(self.rows + self.first_id)


class TestRawDataWrapper_Neuron_H5V1(DataWrapper_Neuron):
    '''Test HDF5 v1 reading'''
    def setup(self):
        self.data = readers.load_h5v1(
            os.path.join(H5V1_PATH, 'Neuron.h5'))
        self.first_id = int(self.data.data_block[0][COLS.ID])
        self.rows = len(self.data.data_block)


class TestRawDataWrapper_Neuron_H5V2(DataWrapper_Neuron):
    '''Test HDF5 v2 reading'''
    def setup(self):
        self.data = readers.load_h5v2(
            os.path.join(H5V2_PATH, 'Neuron.h5'), stage='raw')
        self.first_id = int(self.data.data_block[0][COLS.ID])
        self.rows = len(self.data.data_block)
