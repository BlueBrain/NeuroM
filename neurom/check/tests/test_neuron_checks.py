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
from copy import deepcopy

from nose import tools as nt

from neurom import load_neuron
from neurom import check
from neurom.check import neuron_checks as nrn_chk
from neurom.core.dataformat import COLS

from neurom._compat import range


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
        for node in neurite.iter_sections():
            sec = node.points
            if node.parent is not None:
                sec[0][COLS.R] = node.parent.points[-1][COLS.R] / 2.
            for point_id in range(len(sec) - 1):
                sec[point_id + 1][COLS.R] = sec[point_id][COLS.R] / 2.


def _make_flat(neuron):

    class Flattenizer(object):
        def __call__(self, points):
            points = deepcopy(points)
            points[:, COLS.Z] = 0.;
            return points

    return neuron.transform(Flattenizer())


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
        nt.ok_(nrn_chk.has_axon(n))


def test_has_axon_bad_data():
    files = ['Single_apical.swc',
             'Single_basal.swc',
             ]
    for n in _pick(files):
        nt.ok_(not nrn_chk.has_axon(n))


def test_has_apical_dendrite_good_data():
    files = ['Neuron.swc',
             'Neuron_small_radius.swc',
             'Single_apical.swc',
             'Neuron.h5',
             ]

    for n in _pick(files):
        nt.ok_(nrn_chk.has_apical_dendrite(n))


def test_has_apical_dendrite_bad_data():
    files = ['Single_axon.swc',
             'Single_basal.swc',
             ]

    for n in _pick(files):
        nt.ok_(not nrn_chk.has_apical_dendrite(n))


def test_has_basal_dendrite_good_data():
    files = ['Neuron.swc',
             'Neuron_small_radius.swc',
             'Single_basal.swc',
             'Neuron_2_branch.h5',
             'Neuron.h5',
             ]

    for n in _pick(files):
        nt.ok_(nrn_chk.has_basal_dendrite(n))


def test_has_basal_dendrite_bad_data():
    files = ['Single_axon.swc',
             'Single_apical.swc',
             ]

    for n in _pick(files):
        nt.ok_(not nrn_chk.has_basal_dendrite(n))


def test_has_no_flat_neurites():

    _, n = _load_neuron('Neuron.swc')

    nt.assert_true(nrn_chk.has_no_flat_neurites(n, 1e-6, method='tolerance'))
    nt.assert_true(nrn_chk.has_no_flat_neurites(n, 0.1, method='ratio'))

    n = _make_flat(n)

    nt.assert_false(nrn_chk.has_no_flat_neurites(n, 1e-6, method='tolerance'))
    nt.assert_false(nrn_chk.has_no_flat_neurites(n, 0.1, method='ratio'))


def test_has_all_monotonic_neurites():

    _, n = _load_neuron('Neuron.swc')

    nt.assert_false(nrn_chk.has_all_monotonic_neurites(n))

    _make_monotonic(n)

    nt.assert_true(nrn_chk.has_all_monotonic_neurites(n))


def test_nonzero_neurite_radii_good_data():
    files = ['Neuron.swc',
             'Single_apical.swc',
             'Single_basal.swc',
             'Single_axon.swc',
             'Neuron_2_branch.h5',
             ]

    for n in _pick(files):
        ids = nrn_chk.has_all_nonzero_neurite_radii(n)
        nt.ok_(len(ids.info) == 0)


def test_has_all_nonzero_neurite_radii_threshold():
    nrn = NEURONS['Neuron.swc']

    ids = nrn_chk.has_all_nonzero_neurite_radii(nrn)
    nt.ok_(ids.status)

    ids = nrn_chk.has_all_nonzero_neurite_radii(nrn, threshold=0.25)
    nt.assert_equal(len(ids.info), 122)


def test_nonzero_neurite_radii_bad_data():
    nrn = NEURONS['Neuron_zero_radius.swc']
    ids = nrn_chk.has_all_nonzero_neurite_radii(nrn)
    nt.assert_equal(ids.info, [(20, 10), (21, 0),
                               (22, 0), (22, 6),
                               (26, 1), (31, 9),
                               (50, 7)])


def test_nonzero_segment_lengths_good_data():
    nrn = NEURONS['Neuron.swc']
    ids = nrn_chk.has_all_nonzero_segment_lengths(nrn)
    nt.ok_(ids.status)
    nt.ok_(len(ids.info) == 0)


def test_nonzero_segment_lengths_bad_data():
    files = ['Neuron_zero_length_segments.swc',
             'Single_apical.swc',
             'Single_basal.swc',
             'Single_axon.swc',
             ]

    bad_ids = [[(2, 0), (23, 0), (44, 0), (65, 0)],
               [(2, 0)],
               [(2, 0)],
               [(2, 0)],
               [(2, 0)]]

    for i, nrn in enumerate(_pick(files)):
        ids = nrn_chk.has_all_nonzero_segment_lengths(nrn)
        nt.assert_equal(ids.info, bad_ids[i])


def test_nonzero_segment_lengths_threshold():
    nrn = NEURONS['Neuron.swc']

    ids = nrn_chk.has_all_nonzero_segment_lengths(nrn)
    nt.ok_(ids.status)
    nt.assert_equal(len(ids.info), 0)

    ids = nrn_chk.has_all_nonzero_segment_lengths(nrn, threshold=0.25)
    nt.assert_equal(ids.info, [(2, 0), (23, 0), (38, 9), (44, 0),
                               (54, 7), (62, 2), (65, 0), (72, 4), (78, 6)])


def test_nonzero_section_lengths_good_data():
    files = ['Neuron.swc',
             'Single_apical.swc',
             'Single_basal.swc',
             'Single_axon.swc',
             ]

    for i, nrn in enumerate(_pick(files)):
        ids = nrn_chk.has_all_nonzero_section_lengths(nrn)
        nt.ok_(ids.status)
        nt.ok_(len(ids.info) == 0)


def test_nonzero_section_lengths_bad_data():
    nrn = NEURONS['Neuron_zero_length_sections.swc']

    ids = nrn_chk.has_all_nonzero_section_lengths(nrn)
    nt.ok_(not ids.status)
    nt.assert_equal(ids.info, [15])


def test_nonzero_section_lengths_threshold():
    nrn = NEURONS['Neuron.swc']

    ids = nrn_chk.has_all_nonzero_section_lengths(nrn)
    nt.ok_(ids.status)
    nt.ok_(len(ids.info) == 0)

    ids = nrn_chk.has_all_nonzero_section_lengths(nrn, threshold=15.)
    nt.ok_(not ids.status)
    nt.assert_equal(len(ids.info), 84)


def test_has_nonzero_soma_radius():

    nrn = load_neuron(os.path.join(SWC_PATH, 'Neuron.swc'))
    nt.assert_true(nrn_chk.has_nonzero_soma_radius(nrn))


def test_has_nonzero_soma_radius_bad_data():

    nrn = load_neuron(os.path.join(SWC_PATH, 'Single_basal.swc'))
    nt.assert_false(nrn_chk.has_nonzero_soma_radius(nrn).status)


def test_has_no_fat_ends():
    _, nrn = _load_neuron('fat_end.swc')
    nt.ok_(not nrn_chk.has_no_fat_ends(nrn).status)

    # if we only use point, there isn't a 'fat end'
    # since if the last point is 'x': x < 2*mean([x])
    nt.ok_(nrn_chk.has_no_fat_ends(nrn, final_point_count=1).status)

    # if the multiple of the mean is large, the end won't be fat
    nt.ok_(nrn_chk.has_no_fat_ends(nrn, multiple_of_mean=10).status)

    _, nrn = _load_neuron('Single_basal.swc')
    nt.ok_(nrn_chk.has_no_fat_ends(nrn).status)


def test_has_nonzero_soma_radius_threshold():

    class Dummy(object):
        pass

    nrn = Dummy()
    nrn.soma = Dummy()
    nrn.soma.radius = 1.5

    nt.assert_true(nrn_chk.has_nonzero_soma_radius(nrn))
    nt.assert_true(nrn_chk.has_nonzero_soma_radius(nrn, 0.25))
    nt.assert_true(nrn_chk.has_nonzero_soma_radius(nrn, 0.75))
    nt.assert_true(nrn_chk.has_nonzero_soma_radius(nrn, 1.25))
    nt.assert_true(nrn_chk.has_nonzero_soma_radius(nrn, 1.499))
    nt.assert_false(nrn_chk.has_nonzero_soma_radius(nrn, 1.5))
    nt.assert_false(nrn_chk.has_nonzero_soma_radius(nrn, 1.75))
    nt.assert_false(nrn_chk.has_nonzero_soma_radius(nrn, 2.5))


def test_has_no_jumps():
    _, nrn = _load_neuron('z_jump.swc')
    nt.ok_(not nrn_chk.has_no_jumps(nrn).status)
    nt.ok_(nrn_chk.has_no_jumps(nrn, 100).status)

    nt.ok_(nrn_chk.has_no_jumps(nrn, 100, axis='x').status)

def test__bool__():
    c = check.CheckResult(status=True)
    nt.ok_(c.__nonzero__())
    nt.eq_(c.__bool__(), c.__nonzero__())
