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
from neurom.io.utils import load_neuron
from neurom.check import morphology as check
from neurom.core.dataformat import ROOT_ID
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE
from neurom.analysis.morphtree import get_tree_type
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')
H5V1_PATH = os.path.join(DATA_PATH, 'h5/v1')


def test_has_axon_good_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Neuron_small_radius.swc',
                       'Single_axon.swc']]

    files.append(os.path.join(H5V1_PATH, 'Neuron.h5'))

    neurons = [load_neuron(f, get_tree_type) for f in files]
    for n in neurons:
        nt.ok_(check.has_axon(n))


def test_has_axon_bad_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Single_apical.swc',
                       'Single_basal.swc']]

    neurons = [load_neuron(f, get_tree_type) for f in files]
    for n in neurons:
        nt.ok_(not check.has_axon(n))


def test_has_apical_dendrite_good_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Neuron_small_radius.swc',
                       'Single_apical.swc']]

    files.append(os.path.join(H5V1_PATH, 'Neuron.h5'))

    neurons = [load_neuron(f, get_tree_type) for f in files]
    for n in neurons:
        nt.ok_(check.has_apical_dendrite(n))


def test_has_apical_dendrite_bad_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Single_axon.swc',
                       'Single_basal.swc']]

    neurons = [load_neuron(f, get_tree_type) for f in files]
    for n in neurons:
        nt.ok_(not check.has_apical_dendrite(n))


def test_has_basal_dendrite_good_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Neuron_small_radius.swc',
                       'Single_basal.swc']]

    files.extend([os.path.join(H5V1_PATH, 'Neuron_2_branch.h5'),
                  os.path.join(H5V1_PATH, 'Neuron.h5')])

    neurons = [load_neuron(f, get_tree_type) for f in files]
    for n in neurons:
        nt.ok_(check.has_basal_dendrite(n))


def test_has_basal_dendrite_bad_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Single_axon.swc',
                       'Single_apical.swc']]

    neurons = [load_neuron(f, get_tree_type) for f in files]
    for n in neurons:
        nt.ok_(not check.has_basal_dendrite(n))


def test_all_nonzero_neurite_radii_good_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Single_apical.swc',
                       'Single_basal.swc',
                       'Single_axon.swc']]

    files.append(os.path.join(H5V1_PATH, 'Neuron_2_branch.h5'))

    for f in files:
        ok, ids = check.all_nonzero_neurite_radii(load_neuron(f))
        nt.ok_(ok)
        nt.ok_(len(ids) == 0)


def test_all_nonzero_neurite_radii_threshold():
    mf = os.path.join(SWC_PATH, 'Neuron.swc')
    nrn = load_neuron(mf)

    ok, ids = check.all_nonzero_neurite_radii(nrn)
    nt.ok_(ok)
    nt.ok_(len(ids) == 0)

    ok, ids = check.all_nonzero_neurite_radii(nrn, threshold=0.25)
    nt.ok_(not ok)
    nt.assert_equal(len(ids), 118)


def test_all_nonzero_neurite_radii_bad_data():
    f = os.path.join(SWC_PATH, 'Neuron_zero_radius.swc')
    ok, ids = check.all_nonzero_neurite_radii(load_neuron(f))
    nt.ok_(not ok)
    nt.ok_(ids == [194, 210, 246, 304, 493])


def test_all_nonzero_segment_lengths_good_data():
    f = os.path.join(SWC_PATH, 'Neuron.swc')

    ok, ids = check.all_nonzero_segment_lengths(load_neuron(f))
    nt.ok_(ok)
    nt.ok_(len(ids) == 0)


def test_all_nonzero_segment_lengths_bad_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron_zero_length_segments.swc',
                       'Single_apical.swc',
                       'Single_basal.swc',
                       'Single_axon.swc']]

    bad_segs = [[(4, 5), (215, 216),
                 (426, 427), (637, 638)],
                [(4, 5)],
                [(4, 5)],
                [(4, 5)]]

    for i, f in enumerate(files):
        ok, ids = check.all_nonzero_segment_lengths(load_neuron(f))
        nt.ok_(not ok)
        nt.assert_equal(ids, bad_segs[i])


def test_all_nonzero_segment_lengths_threshold():
    f = os.path.join(SWC_PATH, 'Neuron.swc')
    nrn = load_neuron(f)

    ok, ids = check.all_nonzero_segment_lengths(nrn)
    nt.ok_(ok)
    nt.assert_equal(len(ids), 0)

    ok, ids = check.all_nonzero_segment_lengths(nrn, threshold=0.25)
    nt.ok_(not ok)
    nt.assert_equal(ids, [(4, 5), (215, 216), (374, 375), (426, 427),
                          (533, 534), (608, 609), (637, 638), (711, 712),
                          (773, 774)])


def test_all_nonzero_section_lengths_good_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Single_apical.swc',
                       'Single_basal.swc',
                       'Single_axon.swc']]
    for i, f in enumerate(files):
        ok, ids = check.all_nonzero_section_lengths(load_neuron(f))
        nt.ok_(ok)
        nt.ok_(len(ids) == 0)



@nt.nottest  # TODO We need data sample with a soma and zero length sections
def test_all_nonzero_section_lengths_bad_data():
    files = [os.path.join(SWC_PATH, f)
             for f in []]

    bad_segs = [[]]

    for i, f in enumerate(files):
        ok, ids = check.all_nonzero_section_lengths(load_neuron(f))
        nt.ok_(not ok)
        nt.assert_equal(ids, bad_segs[i])


def test_all_nonzero_section_lengths_threshold():
    mf = os.path.join(SWC_PATH, 'Neuron.swc')
    nrn = load_neuron(mf)

    ok, ids = check.all_nonzero_section_lengths(nrn)
    nt.ok_(ok)
    nt.ok_(len(ids) == 0)

    ok, ids = check.all_nonzero_section_lengths(nrn, threshold=15.)
    nt.ok_(not ok)
    nt.assert_equal(len(ids), 84)
