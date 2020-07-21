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

import h5py
import numpy as np
from neurom.core.dataformat import COLS
from neurom.io import hdf5, swc
from nose import tools as nt

DATA_PATH = Path(__file__).parent.parent.parent.parent / 'test_data'
H5_PATH = Path(DATA_PATH, 'h5')
H5V1_PATH = Path(H5_PATH, 'v1')
H5V2_PATH = Path(H5_PATH, 'v2')
SWC_PATH = Path(DATA_PATH, 'swc')


def test_get_version():
    with h5py.File(Path(H5V1_PATH, 'Neuron.h5'), mode='r') as v1:
        nt.assert_equal(hdf5.get_version(v1), 'H5V1')

    with h5py.File(Path(H5V2_PATH, 'Neuron.h5'), mode='r') as v2:
        nt.assert_equal(hdf5.get_version(v2), 'H5V2')
    nt.assert_equal(hdf5.get_version({}), None)


def test_unpack_h5():
    with h5py.File(Path(H5V1_PATH, 'Neuron.h5'), mode='r') as v1:
        pts1, grp1 = hdf5._unpack_v1(v1)

    with h5py.File(Path(H5V2_PATH, 'Neuron.h5'), mode='r') as v2:
        pts2, grp2 = hdf5._unpack_v2(v2, stage='raw')

    nt.assert_true(np.all(pts1 == pts2))
    nt.assert_true(np.all(grp1 == grp2))


def test_consistency_between_v1_v2():
    v1_data = hdf5.read(Path(H5V1_PATH, 'Neuron.h5'))
    v2_data = hdf5.read(Path(H5V2_PATH, 'Neuron.h5'))
    nt.ok_(np.allclose(v1_data.data_block, v2_data.data_block))


def test_consistency_between_h5_swc():
    h5_data = hdf5.read(Path(H5V1_PATH, 'Neuron.h5'), remove_duplicates=True)
    swc_data = swc.read(Path(SWC_PATH, 'Neuron.swc'))
    nt.ok_(np.allclose(h5_data.data_block.shape, swc_data.data_block.shape))


class DataWrapper_Neuron(object):
    """Base class for H5 tests."""

    end_pts = [1, 775, 393, 524, 142, 655, 273, 22, 795, 413,
               544, 162, 675, 293, 423, 42, 815, 564, 182, 695,
               313, 444, 62, 835, 584, 202, 715, 333, 846, 845,
               464, 82, 212, 604, 634, 735, 353, 484, 102, 233,
               624, 755, 373, 504, 122, 253]

    end_parents = [0, 774, 392, 523, 141, 654, 272, 21, 794, 412,
                   543, 161, 674, 292, 422, 41, 814, 563, 181, 694,
                   312, 443, 61, 834, 583, 201, 714, 332, 0, 844,
                   463, 81, 211, 603, 633, 734, 352, 483, 101, 232,
                   623, 754, 372, 503, 121, 252]

    fork_pts = [0, 12, 32, 52, 72, 92, 112, 132, 152, 172, 192, 223,
                243, 263, 283, 303, 323, 343, 363, 383, 403, 434, 454,
                474, 494, 514, 534, 554, 574, 594, 614, 645, 665, 685,
                705, 725, 745, 765, 785, 805, 825]

    fork_parents = [-1, 11, 31, 51, 71, 91, 111, 131, 151, 171, 191, 222,
                    242, 262, 282, 302, 322, 342, 362, 382, 402, 433, 453,
                    473, 493, 513, 533, 553, 573, 593, 613, 644, 664, 684,
                    704, 724, 744, 764, 784, 804, 824]

    def test_n_rows(self):
        nt.assert_equal(self.rows, 927)

    def test_first_id_0(self):
        nt.ok_(self.first_id == 0)


class TestDataWrapper_Neuron_H5V1(DataWrapper_Neuron):
    """Test HDF5 v1 reading."""

    def setup(self):
        self.data = hdf5.read(Path(H5V1_PATH, 'Neuron.h5'))
        self.first_id = int(self.data.data_block[0][COLS.ID])
        self.rows = len(self.data.data_block)


class TestDataWrapper_Neuron_H5V2(DataWrapper_Neuron):
    """Test HDF5 v2 reading."""

    def setup(self):
        self.data = hdf5.read(Path(H5V2_PATH, 'Neuron.h5'))
        self.first_id = int(self.data.data_block[0][COLS.ID])
        self.rows = len(self.data.data_block)


class DataWrapper_Neuron_with_duplicates(object):
    """Base class for H5 tests."""

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

    def test_end_point_parents(self):
        epar = [self.data.get_parent(i) for i in self.data.get_end_points()]
        nt.assert_equal(epar, DataWrapper_Neuron_with_duplicates.end_parents)
