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

from io import StringIO
from pathlib import Path

from neurom import check, load_morphology
from neurom.check import morphology_checks
from neurom.core.dataformat import COLS
from neurom.core.types import dendrite_filter
from neurom.exceptions import NeuroMError
import pytest
from numpy.testing import assert_array_equal

DATA_PATH = Path(__file__).parent.parent / 'data'
SWC_PATH = DATA_PATH / 'swc'
ASC_PATH = DATA_PATH / 'neurolucida'
H5V1_PATH = DATA_PATH / 'h5/v1'


def _load_morphology(name):
    if name.endswith('.swc'):
        path = SWC_PATH / name
    elif name.endswith('.h5'):
        path = H5V1_PATH / name
    else:
        path = ASC_PATH / name
    return name, load_morphology(path)


NEURONS = dict(
    [
        _load_morphology(n)
        for n in [
            'Neuron.h5',
            'Neuron_2_branch.h5',
            'Neuron.swc',
            'Neuron_small_radius.swc',
            'Neuron_zero_length_sections.swc',
            'Neuron_zero_length_segments.swc',
            'Neuron_zero_radius.swc',
            'Single_apical.swc',
            'Single_axon.swc',
            'Single_basal.swc',
        ]
    ]
)


def _pick(files):
    return [NEURONS[f] for f in files]


def test_has_axon_good_data():
    files = [
        'Neuron.swc',
        'Neuron_small_radius.swc',
        'Single_axon.swc',
        'Neuron.h5',
    ]
    for m in _pick(files):
        assert morphology_checks.has_axon(m)


def test_has_axon_bad_data():
    files = ['Single_apical.swc', 'Single_basal.swc']
    for m in _pick(files):
        assert not morphology_checks.has_axon(m)


def test_has_apical_dendrite_good_data():
    files = ['Neuron.swc', 'Neuron_small_radius.swc', 'Single_apical.swc', 'Neuron.h5']

    for m in _pick(files):
        assert morphology_checks.has_apical_dendrite(m)


def test_has_apical_dendrite_bad_data():
    files = ['Single_axon.swc', 'Single_basal.swc']
    for m in _pick(files):
        assert not morphology_checks.has_apical_dendrite(m)


def test_has_basal_dendrite_good_data():
    files = [
        'Neuron.swc',
        'Neuron_small_radius.swc',
        'Single_basal.swc',
        'Neuron_2_branch.h5',
        'Neuron.h5',
    ]

    for m in _pick(files):
        assert morphology_checks.has_basal_dendrite(m)


def test_has_basal_dendrite_bad_data():
    files = ['Single_axon.swc', 'Single_apical.swc']

    for m in _pick(files):
        assert not morphology_checks.has_basal_dendrite(m)


def test_has_no_flat_neurites():
    _, m = _load_morphology('Neuron.swc')

    assert morphology_checks.has_no_flat_neurites(m, 1e-6, method='tolerance')
    assert morphology_checks.has_no_flat_neurites(m, 0.1, method='ratio')

    _, m = _load_morphology('Neuron-flat.swc')

    assert not morphology_checks.has_no_flat_neurites(m, 1e-6, method='tolerance')
    assert not morphology_checks.has_no_flat_neurites(m, 0.1, method='ratio')


def test_nonzero_neurite_radii_good_data():
    files = [
        'Neuron.swc',
        'Single_apical.swc',
        'Single_basal.swc',
        'Single_axon.swc',
        'Neuron_2_branch.h5',
    ]

    for m in _pick(files):
        ids = morphology_checks.has_all_nonzero_neurite_radii(m)
        assert len(ids.info) == 0


def test_has_all_nonzero_neurite_radii_threshold():
    m = NEURONS['Neuron.swc']

    ids = morphology_checks.has_all_nonzero_neurite_radii(m)
    assert ids.status

    ids = morphology_checks.has_all_nonzero_neurite_radii(m, threshold=0.25)
    assert len(ids.info) == 122


def test_nonzero_neurite_radii_bad_data():
    m = NEURONS['Neuron_zero_radius.swc']
    ids = morphology_checks.has_all_nonzero_neurite_radii(m, threshold=0.7)
    assert ids.info == [(0, 2)]


def test_nonzero_segment_lengths_good_data():
    m = NEURONS['Neuron.swc']
    ids = morphology_checks.has_all_nonzero_segment_lengths(m)
    assert ids.status
    assert len(ids.info) == 0


def test_nonzero_segment_lengths_bad_data():
    files = [
        'Neuron_zero_length_segments.swc',
        'Single_apical.swc',
        'Single_basal.swc',
        'Single_axon.swc',
    ]

    bad_ids = [[0, 21, 42, 63], [0], [0], [0], [0]]

    for i, m in enumerate(_pick(files)):
        ids = morphology_checks.has_all_nonzero_segment_lengths(m)
        assert ids.info == [(id, 0) for id in bad_ids[i]]


def test_nonzero_segment_lengths_threshold():
    m = NEURONS['Neuron.swc']

    ids = morphology_checks.has_all_nonzero_segment_lengths(m)
    assert ids.status
    assert len(ids.info) == 0

    ids = morphology_checks.has_all_nonzero_segment_lengths(m, threshold=0.25)

    bad_ids = [(0, 0), (21, 0), (36, 9), (42, 0), (52, 7), (60, 2), (63, 0), (70, 4), (76, 6)]
    assert ids.info == [(id, val) for id, val in bad_ids]


def test_nonzero_section_lengths_good_data():
    files = [
        'Neuron.swc',
        'Single_apical.swc',
        'Single_basal.swc',
        'Single_axon.swc',
    ]

    for i, m in enumerate(_pick(files)):
        ids = morphology_checks.has_all_nonzero_section_lengths(m)
        assert ids.status
        assert len(ids.info) == 0


def test_nonzero_section_lengths_bad_data():
    m = NEURONS['Neuron_zero_length_sections.swc']

    ids = morphology_checks.has_all_nonzero_section_lengths(m)
    assert not ids.status
    assert ids.info == [13]


def test_nonzero_section_lengths_threshold():
    m = NEURONS['Neuron.swc']

    ids = morphology_checks.has_all_nonzero_section_lengths(m)
    assert ids.status
    assert len(ids.info) == 0

    ids = morphology_checks.has_all_nonzero_section_lengths(m, threshold=15.0)
    assert not ids.status
    assert len(ids.info) == 84


def test_has_nonzero_soma_radius():
    m = load_morphology(SWC_PATH / 'Neuron.swc')
    assert morphology_checks.has_nonzero_soma_radius(m)


def test_has_nonzero_soma_radius_bad_data():
    m = load_morphology(SWC_PATH / 'soma_zero_radius.swc')
    assert not morphology_checks.has_nonzero_soma_radius(m).status


def test_has_no_fat_ends():
    _, m = _load_morphology('fat_end.swc')
    assert not morphology_checks.has_no_fat_ends(m).status

    # if we only use point, there isn't a 'fat end'
    # since if the last point is 'x': x < 2*mean([x])
    assert morphology_checks.has_no_fat_ends(m, final_point_count=1).status

    # if the multiple of the mean is large, the end won't be fat
    assert morphology_checks.has_no_fat_ends(m, multiple_of_mean=10).status

    _, m = _load_morphology('Single_basal.swc')
    assert morphology_checks.has_no_fat_ends(m).status


def test_has_no_root_node_jumps():
    _, m = _load_morphology('root_node_jump.swc')
    check = morphology_checks.has_no_root_node_jumps(m)
    assert not check.status
    assert len(check.info) == 1
    assert check.info[0][0] == 0
    assert_array_equal(check.info[0][1], [[0, 3, 0]])

    assert morphology_checks.has_no_root_node_jumps(m, radius_multiplier=4).status


def test_has_no_narrow_start():
    _, m = _load_morphology('narrow_start.swc')
    check = morphology_checks.has_no_narrow_start(m)
    assert not check.status
    assert_array_equal(check.info[0][1][:, COLS.XYZR], [[0, 0, 2, 2]])

    _, m = _load_morphology('narrow_start.swc')
    assert morphology_checks.has_no_narrow_start(m, 0.25).status

    _, m = _load_morphology('fat_end.swc')  # doesn't have narrow start
    assert morphology_checks.has_no_narrow_start(m, 0.25).status


def test_has_nonzero_soma_radius_threshold():
    class Dummy:
        pass

    m = Dummy()
    m.soma = Dummy()
    m.soma.radius = 1.5

    assert morphology_checks.has_nonzero_soma_radius(m)
    assert morphology_checks.has_nonzero_soma_radius(m, 0.25)
    assert morphology_checks.has_nonzero_soma_radius(m, 0.75)
    assert morphology_checks.has_nonzero_soma_radius(m, 1.25)
    assert morphology_checks.has_nonzero_soma_radius(m, 1.499)
    assert not morphology_checks.has_nonzero_soma_radius(m, 1.5)
    assert not morphology_checks.has_nonzero_soma_radius(m, 1.75)
    assert not morphology_checks.has_nonzero_soma_radius(m, 2.5)


def test_has_no_jumps():
    _, m = _load_morphology('z_jump.swc')
    assert not morphology_checks.has_no_jumps(m).status
    assert morphology_checks.has_no_jumps(m, 100).status

    assert morphology_checks.has_no_jumps(m, 100, axis='x').status


def test_has_no_narrow_dendritic_section():
    swc_content = StringIO(
        u"""
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
"""
    )
    m = load_morphology(swc_content, reader='swc')
    res = morphology_checks.has_no_narrow_neurite_section(
        m, dendrite_filter, radius_threshold=5, considered_section_min_length=0
    )

    assert res.status

    res = morphology_checks.has_no_narrow_neurite_section(
        m, dendrite_filter, radius_threshold=7, considered_section_min_length=0
    )
    assert not res.status

    swc_content = StringIO(
        u"""
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
"""
    )
    m = load_morphology(swc_content, reader='swc')
    res = morphology_checks.has_no_narrow_neurite_section(
        m, dendrite_filter, radius_threshold=5, considered_section_min_length=0
    )
    assert (
        res.status
    ), 'Narrow soma or axons should not raise bad status when checking for narrow dendrites'


def test_has_no_dangling_branch():
    _, m = _load_morphology('dangling_axon.swc')
    res = morphology_checks.has_no_dangling_branch(m)
    assert not res.status
    assert len(res.info) == 1
    assert_array_equal(res.info[0][1][0][COLS.XYZ], [0.0, 49.0, 0.0])

    _, m = _load_morphology('dangling_dendrite.swc')
    res = morphology_checks.has_no_dangling_branch(m)
    assert not res.status
    assert len(res.info) == 1
    assert_array_equal(res.info[0][1][0][COLS.XYZ], [0.0, 49.0, 0.0])

    _, m = _load_morphology('axon-sprout-from-dendrite.asc')
    res = morphology_checks.has_no_dangling_branch(m)
    assert res.status


def test_dangling_branch_no_soma():
    with pytest.raises(NeuroMError, match='Can\'t check for dangling neurites if there is no soma'):
        m = load_morphology(SWC_PATH / 'Single_apical_no_soma.swc')
        morphology_checks.has_no_dangling_branch(m)


def test__bool__():
    c = check.CheckResult(status=True)
    assert c.__nonzero__()
    assert c.__bool__() == c.__nonzero__()


def test_has_multifurcation():
    m = load_morphology(
        StringIO(
            u"""
	((CellBody) (-1 0 0 2) (1 0 0 2))
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
"""
        ),
        reader='asc',
    )

    check_ = morphology_checks.has_multifurcation(m)
    assert not check_.status
    info = check_.info
    assert_array_equal(info[0][0], 0)
    assert_array_equal(info[0][1][:, COLS.XYZR], [[0.0, 13.0, 0.0, 1.0]])


def test_has_unifurcation():
    m = load_morphology(
        StringIO(
            u"""
((CellBody) (-1 0 0 2) (1 0 0 2))

 ((Dendrite)
  (0 0 0 2)
  (0 5 0 2)
  (
   (-5 5 0 3)
   (
    (-10 5 0 3)
   )
   |
   (6 5 0 3)
   )
  )
"""
        ),
        reader='asc',
    )

    check_ = morphology_checks.has_unifurcation(m)
    assert not check_.status
    info = check_.info
    assert_array_equal(info[0][0], 1)
    assert_array_equal(info[0][1][:, COLS.XYZR], [[-5.0, 5.0, 0.0, 1.5]])


def test_single_children():
    m = load_morphology(
        """
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
""",
        "asc",
    )
    result = morphology_checks.has_no_single_children(m)
    assert result.status is False
    assert result.info == [0]


def test_has_no_back_tracking():
    m = load_morphology(
        """
    ((CellBody) (-1 0 0 2) (1 0 0 2))

    ((Dendrite)
    (0 0 0 0.4)
    (0 1 0 0.3)
    (0 2 0 0.28)
    (
      (0 2 0 0.28)
      (1 3 0 0.3)
      (2 4 0 0.22)
      |
      (0 2 0 0.28)
      (1 -3 0 0.3)
      (2 -4 0 0.24)
      (1 -3 0 0.52)
      (3 -5 0 0.2)
      (4 -6 0 0.2)
    ))
""",
        "asc",
    )
    result = morphology_checks.has_no_back_tracking(m)
    assert result.status is False
    info = result.info
    assert_array_equal(info[0][0], [2, 1, 0])
    assert_array_equal(info[0][1], [[1, -3, 0]])
    assert_array_equal(info[1][0], [2, 1, 1])
    assert_array_equal(info[1][1], [[1, -3, 0]])


def test_has_no_overlapping_point():
    m = load_morphology(
        """
    ((CellBody) (-1 0 0 2) (1 0 0 2))

    ((Dendrite)
    (0 0 0 0.4)
    (0 1 0 0.3)
    (0 2 0 0.28)
    (
      (0 2 0 0.28)
      (1 3 0 0.3)
      (2 4 0 0.22)
      |
      (0 2 0 0.28)
      (1 -3 0 0.3)
      (2 -4 0 0.24)
      (1 -3 0 0.52)
      (0  1 0 0.2)
      (4 -6 0 0.2)
    ))
""",
        "asc",
    )
    result = morphology_checks.has_no_overlapping_point(m)
    assert result.status is False
    info = result.info
    assert_array_equal(info[0][0], [0, 2])
    assert_array_equal(info[0][1], [[0, 1, 0]])
    assert_array_equal(info[1][0], [2, 2])
    assert_array_equal(info[1][1], [[1, -3, 0]])
