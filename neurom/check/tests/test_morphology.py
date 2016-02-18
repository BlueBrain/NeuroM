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
from neurom.core.tree import ipreorder
from neurom.io.utils import load_neuron
from neurom.check import morphology as check_morph
from neurom.core.dataformat import COLS
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')
H5V1_PATH = os.path.join(DATA_PATH, 'h5/v1')


def _load_neuron(name):
    if name.endswith('.swc'):
        path = os.path.join(SWC_PATH, name)
    elif name.endswith('.h5'):
        path = os.path.join(H5V1_PATH, name)
    return name, load_neuron(path)


def _make_monotonic(neuron):
    for neurite in neuron.neurites:
        for node in ipreorder(neurite):
            if node.parent is not None:
                node.value[COLS.R] = node.parent.value[COLS.R] / 2.


def _make_flat(neuron):
    for neurite in neuron.neurites:
        for node in ipreorder(neurite):
            if node.parent is not None:
                node.value[COLS.Z] = 0.

NEURONS = dict([_load_neuron(n) for n in ['Neuron.h5',
                                          'Neuron_2_branch.h5',
                                          'Neuron.swc',
                                          'Neuron_small_radius.swc',
                                          'Neuron_zero_length_sections.swc',
                                          'Neuron_zero_length_segments.swc',
                                          'Neuron_zero_radius.swc',
                                          'Single_apical.swc',
                                          'Single_axon.swc',
                                          'Single_basal.swc',
                                          ]])

def _pick(files):
    return [NEURONS[f] for f in files]


def test_has_axon_good_data():
    files = ['Neuron.swc',
             'Neuron_small_radius.swc',
             'Single_axon.swc',
             'Neuron.h5',
             ]
    for n in _pick(files):
        nt.ok_(check_morph.has_axon(n))


def test_has_axon_bad_data():
    files = ['Single_apical.swc',
             'Single_basal.swc',
             ]

    for n in _pick(files):
        nt.ok_(not check_morph.has_axon(n))


def test_has_apical_dendrite_good_data():
    files = ['Neuron.swc',
             'Neuron_small_radius.swc',
             'Single_apical.swc',
             'Neuron.h5',
             ]

    for n in _pick(files):
        nt.ok_(check_morph.has_apical_dendrite(n))


def test_has_apical_dendrite_bad_data():
    files = ['Single_axon.swc',
             'Single_basal.swc',
             ]

    for n in _pick(files):
        nt.ok_(not check_morph.has_apical_dendrite(n))


def test_has_basal_dendrite_good_data():
    files = ['Neuron.swc',
             'Neuron_small_radius.swc',
             'Single_basal.swc',
             'Neuron_2_branch.h5',
             'Neuron.h5',
             ]

    for n in _pick(files):
        nt.ok_(check_morph.has_basal_dendrite(n))


def test_has_basal_dendrite_bad_data():
    files = ['Single_axon.swc',
             'Single_apical.swc',
             ]

    for n in _pick(files):
        nt.ok_(not check_morph.has_basal_dendrite(n))


def test_get_flat_neurites():

    _, n = _load_neuron('Neuron.swc')

    nt.assert_equal(len(check_morph.get_flat_neurites(n, 1e-6, method='tolerance')), 0)
    nt.assert_equal(len(check_morph.get_flat_neurites(n, 0.1, method='ratio')), 0)

    _make_flat(n)

    nt.assert_equal(len(check_morph.get_flat_neurites(n, 1e-6, method='tolerance')), 4)
    nt.assert_equal(len(check_morph.get_flat_neurites(n, 0.1, method='ratio')), 4)


def test_has_no_flat_neurites():

    _, n = _load_neuron('Neuron.swc')

    nt.assert_true(check_morph.has_no_flat_neurites(n, 1e-6, method='tolerance'))
    nt.assert_true(check_morph.has_no_flat_neurites(n, 0.1, method='ratio'))

    _make_flat(n)

    nt.assert_false(check_morph.has_no_flat_neurites(n, 1e-6, method='tolerance'))
    nt.assert_false(check_morph.has_no_flat_neurites(n, 0.1, method='ratio'))


def test_has_all_monotonic_neurites():

    _, n = _load_neuron('Neuron.swc')

    nt.assert_false(check_morph.has_all_monotonic_neurites(n))

    _make_monotonic(n)

    nt.assert_true(check_morph.has_all_monotonic_neurites(n))


def test_get_nonmonotonic_neurites():

    _, n = _load_neuron('Neuron.swc')

    nt.assert_equal(len(check_morph.get_nonmonotonic_neurites(n)), 4)

    _make_monotonic(n)

    nt.assert_equal(len(check_morph.get_nonmonotonic_neurites(n)), 0)


def test_get_zigzagging_neurites():

    _, n = _load_neuron('Neuron.swc')
    nt.assert_equal(len(check_morph.get_zigzagging_neurites(n)), 4)


def test_nonzero_neurite_radii_good_data():
    files = ['Neuron.swc',
             'Single_apical.swc',
             'Single_basal.swc',
             'Single_axon.swc',
             'Neuron_2_branch.h5',
             ]

    for n in _pick(files):
        ids = check_morph.nonzero_neurite_radii(n)
        nt.ok_(len(ids) == 0)


def test_nonzero_neurite_radii_threshold():
    nrn = NEURONS['Neuron.swc']

    ids = check_morph.nonzero_neurite_radii(nrn)
    nt.ok_(len(ids) == 0)

    ids = check_morph.nonzero_neurite_radii(nrn, threshold=0.25)
    nt.assert_equal(len(ids), 118)


def test_nonzero_neurite_radii_bad_data():
    nrn = NEURONS['Neuron_zero_radius.swc']
    ids = check_morph.nonzero_neurite_radii(nrn)
    nt.ok_(ids == [194, 210, 246, 304, 493])


def test_nonzero_segment_lengths_good_data():
    nrn = NEURONS['Neuron.swc']
    ids = check_morph.nonzero_segment_lengths(nrn)
    nt.ok_(len(ids) == 0)


def test_nonzero_segment_lengths_bad_data():
    files = ['Neuron_zero_length_segments.swc',
             'Single_apical.swc',
             'Single_basal.swc',
             'Single_axon.swc',
             ]

    bad_segs = [[(4, 5), (215, 216),
                 (426, 427), (637, 638)],
                [(4, 5)],
                [(4, 5)],
                [(4, 5)]]

    for i, nrn in enumerate(_pick(files)):
        ids = check_morph.nonzero_segment_lengths(nrn)
        nt.assert_equal(ids, bad_segs[i])


def test_nonzero_segment_lengths_threshold():
    nrn = NEURONS['Neuron.swc']

    ids = check_morph.nonzero_segment_lengths(nrn)
    nt.assert_equal(len(ids), 0)

    ids = check_morph.nonzero_segment_lengths(nrn, threshold=0.25)
    nt.assert_equal(ids, [(4, 5), (215, 216), (374, 375), (426, 427),
                          (533, 534), (608, 609), (637, 638), (711, 712),
                          (773, 774)])


def test_nonzero_section_lengths_good_data():
    files = ['Neuron.swc',
             'Single_apical.swc',
             'Single_basal.swc',
             'Single_axon.swc',
             ]

    for i, nrn in enumerate(_pick(files)):
        ids = check_morph.nonzero_section_lengths(nrn)
        nt.ok_(len(ids) == 0)


def test_nonzero_section_lengths_bad_data():
    nrn = NEURONS['Neuron_zero_length_sections.swc']

    ids = check_morph.nonzero_section_lengths(nrn)
    nt.assert_equal(ids, [134])


def test_nonzero_section_lengths_threshold():
    nrn = NEURONS['Neuron.swc']

    ids = check_morph.nonzero_section_lengths(nrn)
    nt.ok_(len(ids) == 0)

    ids = check_morph.nonzero_section_lengths(nrn, threshold=15.)
    nt.assert_equal(len(ids), 84)
