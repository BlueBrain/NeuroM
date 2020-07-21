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
from copy import deepcopy
from io import StringIO

from nose import tools as nt
from nose.tools import assert_equal
from numpy.testing import assert_array_equal

from neurom import check, load_neuron
from neurom.check import neuron_checks as nrn_chk
from neurom.core.dataformat import COLS
from neurom.core.types import dendrite_filter

DATA_PATH = Path(__file__).parent.parent.parent.parent / 'test_data'
SWC_PATH = Path(DATA_PATH, 'swc')
ASC_PATH = Path(DATA_PATH, 'neurolucida')
H5V1_PATH = Path(DATA_PATH, 'h5/v1')


def _load_neuron(name):
    if name.endswith('.swc'):
        path = Path(SWC_PATH, name)
    elif name.endswith('.h5'):
        path = Path(H5V1_PATH, name)
    else:
        path = Path(ASC_PATH, name)
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

    nrn = load_neuron(Path(SWC_PATH, 'Neuron.swc'))
    nt.assert_true(nrn_chk.has_nonzero_soma_radius(nrn))


def test_has_nonzero_soma_radius_bad_data():

    nrn = load_neuron(Path(SWC_PATH, 'Single_basal.swc'))
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


def test_has_no_root_node_jumps():
    _, nrn = _load_neuron('root_node_jump.swc')
    check = nrn_chk.has_no_root_node_jumps(nrn)
    nt.ok_(not check.status)
    assert_equal(len(check.info), 1)
    assert_equal(check.info[0][0], 1)
    assert_array_equal(check.info[0][1], [[0, 3, 0]])

    nt.ok_(nrn_chk.has_no_root_node_jumps(nrn, radius_multiplier=4).status)


def test_has_no_narrow_start():
    _, nrn = _load_neuron('narrow_start.swc')
    check = nrn_chk.has_no_narrow_start(nrn)
    nt.ok_(not check.status)
    assert_array_equal(check.info[0][1][:, COLS.XYZR], [[0, 0, 2, 2]])

    _, nrn = _load_neuron('narrow_start.swc')
    nt.ok_(nrn_chk.has_no_narrow_start(nrn, 0.25).status)

    _, nrn = _load_neuron('fat_end.swc')  # doesn't have narrow start
    nt.ok_(nrn_chk.has_no_narrow_start(nrn, 0.25).status)


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


def test_has_no_narrow_dendritic_section():
    swc_content = StringIO(u"""
# index, type, x, y, z, radius, parent
    1 1  0  0 0 10. -1
    2 2  0  0 0 10.  1
    3 2  0  50 0 10.  2
    4 2 -5  51 0 10.  3
    5 2  6  53 0 10.  3
    6 3  0  0 0 5.  1  # start of the narrow section
    7 3  0 -4 0 5.  6
    8 3  6 -4 0 10.  7
    9 3 -5 -4 0 10.  7
""")
    nrn = load_neuron(swc_content, reader='swc')
    res = nrn_chk.has_no_narrow_neurite_section(nrn,
                                                dendrite_filter,
                                                radius_threshold=5,
                                                considered_section_min_length=0)

    nt.ok_(res.status)

    res = nrn_chk.has_no_narrow_neurite_section(nrn, dendrite_filter,
                                                radius_threshold=7,
                                                considered_section_min_length=0)
    nt.ok_(not res.status)

    swc_content = StringIO(u"""
# index, type, x, y, z, radius, parent
    1 1  0  0 0 10. -1
    2 2  0  0 0 5  1 # narrow soma
    3 2  0  50 0 5  2
    4 2 -5  51 0 5  3
    5 2  6  53 0 5  3
    6 3  0  0 0 5  1 # narrow axon
    7 3  0 -4 0 10.  6
    8 3  6 -4 0 10.  7
    9 3 -5 -4 0 10.  7
""")
    res = nrn_chk.has_no_narrow_neurite_section(nrn, dendrite_filter,
                                                radius_threshold=5,
                                                considered_section_min_length=0)
    nt.ok_(res.status, 'Narrow soma or axons should not raise bad status when checking for narrow dendrites')


def test_has_no_dangling_branch():
    _, nrn = _load_neuron('dangling_axon.swc')
    res = nrn_chk.has_no_dangling_branch(nrn)
    nt.ok_(not res.status)
    nt.assert_equal(len(res.info), 1)
    assert_array_equal(res.info[0][1][0][COLS.XYZ],
                       [0., 49.,  0.])

    _, nrn = _load_neuron('dangling_dendrite.swc')
    res = nrn_chk.has_no_dangling_branch(nrn)
    nt.ok_(not res.status)
    nt.assert_equal(len(res.info), 1)
    assert_array_equal(res.info[0][1][0][COLS.XYZ],
                       [0., 49.,  0.])

    _, nrn = _load_neuron('axon-sprout-from-dendrite.asc')
    res = nrn_chk.has_no_dangling_branch(nrn)
    nt.ok_(res.status)



def test__bool__():
    c = check.CheckResult(status=True)
    nt.ok_(c.__nonzero__())
    nt.eq_(c.__bool__(), c.__nonzero__())



def test_has_multifurcation():
    nrn = load_neuron(StringIO(u"""
	((CellBody) (0 0 0 2))
( (Color Blue)
  (Axon)
  (0 5 0 2)
  (2 9 0 2)
  (0 13 0 2)
  (
    (0 13 0 2)
    (4 13 0 2)
    |
    (0 13 0 2)
    (4 13 0 2)
    |
    (0 13 0 2)
    (4 13 0 2)
    |
    (0 13 0 2)
    (4 13 0 2)
  )
)
"""), reader='asc')

    check_ = nrn_chk.has_multifurcation(nrn)
    nt.ok_(not check_.status)
    info = check_.info
    assert_array_equal(info[0][0], 1)
    assert_array_equal(info[0][1][:, COLS.XYZR], [[0.0, 13.0, 0.0, 1.0]])
