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

from copy import deepcopy
from io import StringIO
from pathlib import Path

from neurom import check, load_neuron
from neurom.check import neuron_checks as nrn_chk
from neurom.core.dataformat import COLS
from neurom.core.types import dendrite_filter
from numpy.testing import assert_array_equal

DATA_PATH = Path(__file__).parent.parent / 'data'
SWC_PATH = DATA_PATH / 'swc'
ASC_PATH = DATA_PATH / 'neurolucida'
H5V1_PATH = DATA_PATH / 'h5/v1'



def _load_neuron(name):
    if name.endswith('.swc'):
        path = SWC_PATH / name
    elif name.endswith('.h5'):
        path = H5V1_PATH / name
    else:
        path = ASC_PATH / name
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

    class Flattenizer:
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
        assert nrn_chk.has_axon(n)


def test_has_axon_bad_data():
    files = ['Single_apical.swc',
             'Single_basal.swc',
             ]
    for n in _pick(files):
        assert not nrn_chk.has_axon(n)


def test_has_apical_dendrite_good_data():
    files = ['Neuron.swc',
             'Neuron_small_radius.swc',
             'Single_apical.swc',
             'Neuron.h5',
             ]

    for n in _pick(files):
        assert nrn_chk.has_apical_dendrite(n)


def test_has_apical_dendrite_bad_data():
    files = ['Single_axon.swc',
             'Single_basal.swc',
             ]

    for n in _pick(files):
        assert not nrn_chk.has_apical_dendrite(n)


def test_has_basal_dendrite_good_data():
    files = ['Neuron.swc',
             'Neuron_small_radius.swc',
             'Single_basal.swc',
             'Neuron_2_branch.h5',
             'Neuron.h5',
             ]

    for n in _pick(files):
        assert nrn_chk.has_basal_dendrite(n)


def test_has_basal_dendrite_bad_data():
    files = ['Single_axon.swc',
             'Single_apical.swc',
             ]

    for n in _pick(files):
        assert not nrn_chk.has_basal_dendrite(n)


def test_has_no_flat_neurites():

    _, n = _load_neuron('Neuron.swc')

    assert nrn_chk.has_no_flat_neurites(n, 1e-6, method='tolerance')
    assert nrn_chk.has_no_flat_neurites(n, 0.1, method='ratio')

    n = _make_flat(n)

    assert not nrn_chk.has_no_flat_neurites(n, 1e-6, method='tolerance')
    assert not nrn_chk.has_no_flat_neurites(n, 0.1, method='ratio')


def test_nonzero_neurite_radii_good_data():
    files = ['Neuron.swc',
             'Single_apical.swc',
             'Single_basal.swc',
             'Single_axon.swc',
             'Neuron_2_branch.h5',
             ]

    for n in _pick(files):
        ids = nrn_chk.has_all_nonzero_neurite_radii(n)
        assert len(ids.info) == 0


def test_has_all_nonzero_neurite_radii_threshold():
    nrn = NEURONS['Neuron.swc']

    ids = nrn_chk.has_all_nonzero_neurite_radii(nrn)
    assert ids.status

    ids = nrn_chk.has_all_nonzero_neurite_radii(nrn, threshold=0.25)
    assert len(ids.info) == 122


def test_nonzero_neurite_radii_bad_data():
    nrn = NEURONS['Neuron_zero_radius.swc']
    ids = nrn_chk.has_all_nonzero_neurite_radii(nrn, threshold=0.7)
    assert ids.info == [(0, 2)]


def test_nonzero_segment_lengths_good_data():
    nrn = NEURONS['Neuron.swc']
    ids = nrn_chk.has_all_nonzero_segment_lengths(nrn)
    assert ids.status
    assert len(ids.info) == 0


def test_nonzero_segment_lengths_bad_data():
    files = ['Neuron_zero_length_segments.swc',
             'Single_apical.swc',
             'Single_basal.swc',
             'Single_axon.swc',
             ]

    bad_ids = [[0, 21, 42, 63], [0], [0], [0], [0]]

    for i, nrn in enumerate(_pick(files)):
        ids = nrn_chk.has_all_nonzero_segment_lengths(nrn)
        assert (ids.info ==
                        [(id, 0) for id in bad_ids[i]])


def test_nonzero_segment_lengths_threshold():
    nrn = NEURONS['Neuron.swc']

    ids = nrn_chk.has_all_nonzero_segment_lengths(nrn)
    assert ids.status
    assert len(ids.info) == 0

    ids = nrn_chk.has_all_nonzero_segment_lengths(nrn, threshold=0.25)

    bad_ids = [(0, 0), (21, 0), (36, 9), (42, 0), (52, 7), (60, 2), (63, 0), (70, 4), (76, 6)]
    assert (ids.info ==
                    [(id, val) for id, val in bad_ids])


def test_nonzero_section_lengths_good_data():
    files = ['Neuron.swc',
             'Single_apical.swc',
             'Single_basal.swc',
             'Single_axon.swc',
             ]

    for i, nrn in enumerate(_pick(files)):
        ids = nrn_chk.has_all_nonzero_section_lengths(nrn)
        assert ids.status
        assert len(ids.info) == 0


def test_nonzero_section_lengths_bad_data():
    nrn = NEURONS['Neuron_zero_length_sections.swc']

    ids = nrn_chk.has_all_nonzero_section_lengths(nrn)
    assert not ids.status
    assert ids.info == [13]


def test_nonzero_section_lengths_threshold():
    nrn = NEURONS['Neuron.swc']

    ids = nrn_chk.has_all_nonzero_section_lengths(nrn)
    assert ids.status
    assert len(ids.info) == 0

    ids = nrn_chk.has_all_nonzero_section_lengths(nrn, threshold=15.)
    assert not ids.status
    assert len(ids.info) == 84


def test_has_nonzero_soma_radius():

    nrn = load_neuron(SWC_PATH / 'Neuron.swc')
    assert nrn_chk.has_nonzero_soma_radius(nrn)


def test_has_nonzero_soma_radius_bad_data():
    nrn = load_neuron(SWC_PATH / 'soma_zero_radius.swc')
    assert not nrn_chk.has_nonzero_soma_radius(nrn).status


def test_has_no_fat_ends():
    _, nrn = _load_neuron('fat_end.swc')
    assert not nrn_chk.has_no_fat_ends(nrn).status

    # if we only use point, there isn't a 'fat end'
    # since if the last point is 'x': x < 2*mean([x])
    assert nrn_chk.has_no_fat_ends(nrn, final_point_count=1).status

    # if the multiple of the mean is large, the end won't be fat
    assert nrn_chk.has_no_fat_ends(nrn, multiple_of_mean=10).status

    _, nrn = _load_neuron('Single_basal.swc')
    assert nrn_chk.has_no_fat_ends(nrn).status


def test_has_no_root_node_jumps():
    _, nrn = _load_neuron('root_node_jump.swc')
    check = nrn_chk.has_no_root_node_jumps(nrn)
    assert not check.status
    assert len(check.info) == 1
    assert check.info[0][0] == 0
    assert_array_equal(check.info[0][1], [[0, 3, 0]])

    assert nrn_chk.has_no_root_node_jumps(nrn, radius_multiplier=4).status


def test_has_no_narrow_start():
    _, nrn = _load_neuron('narrow_start.swc')
    check = nrn_chk.has_no_narrow_start(nrn)
    assert not check.status
    assert_array_equal(check.info[0][1][:, COLS.XYZR], [[0, 0, 2, 2]])

    _, nrn = _load_neuron('narrow_start.swc')
    assert nrn_chk.has_no_narrow_start(nrn, 0.25).status

    _, nrn = _load_neuron('fat_end.swc')  # doesn't have narrow start
    assert nrn_chk.has_no_narrow_start(nrn, 0.25).status


def test_has_nonzero_soma_radius_threshold():

    class Dummy:
        pass

    nrn = Dummy()
    nrn.soma = Dummy()
    nrn.soma.radius = 1.5

    assert nrn_chk.has_nonzero_soma_radius(nrn)
    assert nrn_chk.has_nonzero_soma_radius(nrn, 0.25)
    assert nrn_chk.has_nonzero_soma_radius(nrn, 0.75)
    assert nrn_chk.has_nonzero_soma_radius(nrn, 1.25)
    assert nrn_chk.has_nonzero_soma_radius(nrn, 1.499)
    assert not nrn_chk.has_nonzero_soma_radius(nrn, 1.5)
    assert not nrn_chk.has_nonzero_soma_radius(nrn, 1.75)
    assert not nrn_chk.has_nonzero_soma_radius(nrn, 2.5)


def test_has_no_jumps():
    _, nrn = _load_neuron('z_jump.swc')
    assert not nrn_chk.has_no_jumps(nrn).status
    assert nrn_chk.has_no_jumps(nrn, 100).status

    assert nrn_chk.has_no_jumps(nrn, 100, axis='x').status


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

    assert res.status

    res = nrn_chk.has_no_narrow_neurite_section(nrn, dendrite_filter,
                                                radius_threshold=7,
                                                considered_section_min_length=0)
    assert not res.status

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
    nrn = load_neuron(swc_content, reader='swc')
    res = nrn_chk.has_no_narrow_neurite_section(nrn, dendrite_filter,
                                                radius_threshold=5,
                                                considered_section_min_length=0)
    assert res.status, 'Narrow soma or axons should not raise bad status when checking for narrow dendrites'


def test_has_no_dangling_branch():
    _, nrn = _load_neuron('dangling_axon.swc')
    res = nrn_chk.has_no_dangling_branch(nrn)
    assert not res.status
    assert len(res.info) == 1
    assert_array_equal(res.info[0][1][0][COLS.XYZ],
                       [0., 49.,  0.])

    _, nrn = _load_neuron('dangling_dendrite.swc')
    res = nrn_chk.has_no_dangling_branch(nrn)
    assert not res.status
    assert len(res.info) == 1
    assert_array_equal(res.info[0][1][0][COLS.XYZ],
                       [0., 49.,  0.])

    _, nrn = _load_neuron('axon-sprout-from-dendrite.asc')
    res = nrn_chk.has_no_dangling_branch(nrn)
    assert res.status


def test__bool__():
    c = check.CheckResult(status=True)
    assert c.__nonzero__()
    assert c.__bool__() == c.__nonzero__()



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
    assert not check_.status
    info = check_.info
    assert_array_equal(info[0][0], 0)
    assert_array_equal(info[0][1][:, COLS.XYZR], [[0.0, 13.0, 0.0, 1.0]])


def test_single_children():
    neuron = load_neuron("""
( (Color Blue)
  (Axon)
  (0 5 0 2)
  (2 9 0 2)
  (0 13 0 2)
  (
    (2 13 0 2)
    (4 13 0 2)
    (6 13 0 2)
  )
)
""", "asc")
    result = nrn_chk.has_no_single_children(neuron)
    assert result.status == False
    assert result.info == [0]
