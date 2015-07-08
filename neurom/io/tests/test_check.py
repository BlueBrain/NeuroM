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
from neurom.io.readers import load_data
from neurom.io import check
from neurom.core.dataformat import ROOT_ID
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')
H5V1_PATH = os.path.join(DATA_PATH, 'h5/v1')


def test_has_sequential_ids_good_data():

    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Single_apical_no_soma.swc',
                       'Single_apical.swc',
                       'Single_basal.swc',
                       'Single_axon.swc',
                       'Neuron_zero_radius.swc',
                       'sequential_trunk_off_0_16pt.swc',
                       'sequential_trunk_off_1_16pt.swc',
                       'sequential_trunk_off_42_16pt.swc',
                       'Neuron_no_missing_ids_no_zero_segs.swc']
             ]

    for f in files:
        ok, ids = check.has_sequential_ids(load_data(f))
        nt.ok_(ok)
        nt.ok_(len(ids) == 0)



def test_has_sequential_ids_bad_data():

    f = os.path.join(SWC_PATH, 'Neuron_missing_ids.swc')

    ok, ids = check.has_sequential_ids(load_data(f))
    nt.ok_(not ok)
    nt.ok_(ids == [6, 217, 428, 639])


def test_has_soma_good_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Single_apical.swc',
                       'Single_basal.swc',
                       'Single_axon.swc']]

    files.append(os.path.join(H5V1_PATH, 'Neuron_2_branch.h5'))

    for f in files:
        nt.ok_(check.has_soma(load_data(f)))


def test_has_soma_bad_data():
    f = os.path.join(SWC_PATH, 'Single_apical_no_soma.swc')
    nt.ok_(not check.has_soma(load_data(f)))


def test_has_finite_radius_neurites_good_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Single_apical.swc',
                       'Single_basal.swc',
                       'Single_axon.swc']]

    files.append(os.path.join(H5V1_PATH, 'Neuron_2_branch.h5'))

    for f in files:
        ok, ids = check.has_all_finite_radius_neurites(load_data(f))
        nt.ok_(ok)
        nt.ok_(len(ids) == 0)


def test_has_finite_radius_neurites_bad_data():
    f = os.path.join(SWC_PATH, 'Neuron_zero_radius.swc')
    ok, ids = check.has_all_finite_radius_neurites(load_data(f))
    nt.ok_(not ok)
    nt.ok_(ids == [194, 210, 246, 304, 493])


def test_has_finite_length_segments_good_data():
    files = [os.path.join(SWC_PATH, f)
             for f in [
                       'sequential_trunk_off_0_16pt.swc',
                       'sequential_trunk_off_1_16pt.swc',
                       'sequential_trunk_off_42_16pt.swc']]
    for f in files:
        ok, ids = check.has_all_finite_length_segments(load_data(f))
        nt.ok_(ok)
        nt.ok_(len(ids) == 0)


def test_has_finite_length_segments_bad_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Single_apical.swc',
                       'Single_basal.swc',
                       'Single_axon.swc']]

    bad_segs = [[(4, 5), (215, 216),
                 (426, 427), (637, 638)],
                [(4, 5)],
                [(4, 5)],
                [(4, 5)]]

    for i, f in enumerate(files):
        ok, ids = check.has_all_finite_length_segments(load_data(f))
        nt.ok_(not ok)
        nt.assert_equal(ids, bad_segs[i])
