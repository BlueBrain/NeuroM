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

"""Test features.neuritefunc."""

from math import pi, sqrt
from pathlib import Path
from unittest.mock import patch

import neurom as nm
import numpy as np
import scipy
from neurom.features import neurite, morphology
from neurom.geom import convex_hull

import pytest
from numpy.testing import assert_allclose

DATA_PATH = Path(__file__).parent.parent / 'data'
H5_PATH = DATA_PATH / 'h5/v1'
SWC_PATH = DATA_PATH / 'swc'
SIMPLE = nm.load_morphology(SWC_PATH / 'simple.swc')
NRN = nm.load_morphology(H5_PATH / 'Neuron.h5')


def test_number_of_bifurcations():
    assert neurite.number_of_bifurcations(SIMPLE.neurites[0]) == 1
    assert neurite.number_of_bifurcations(SIMPLE.neurites[1]) == 1


def test_number_of_forking_points():
    assert neurite.number_of_forking_points(SIMPLE.neurites[0]) == 1
    assert neurite.number_of_forking_points(SIMPLE.neurites[1]) == 1


def test_number_of_leaves():
    assert neurite.number_of_leaves(SIMPLE.neurites[0]) == 2
    assert neurite.number_of_leaves(SIMPLE.neurites[1]) == 2


def test_neurite_volume_density():
    vol = np.array(morphology.total_volume_per_neurite(NRN))
    hull_vol = np.array([convex_hull(n).volume for n in nm.iter_neurites(NRN)])

    vol_density = [neurite.volume_density(s) for s in NRN.neurites]
    assert len(vol_density) == 4
    assert np.allclose(vol_density, vol / hull_vol)

    ref_density = [0.43756606998299519, 0.52464681266899216,
                   0.24068543213643726, 0.26289304906104355]
    assert_allclose(vol_density, ref_density)


def test_neurite_volume_density_failed_convex_hull():

    flat_neuron = nm.load_morphology(
    """
    1  1   0  0  0  0.5 -1
    2  3   1  0  0  0.1  1
    3  3   2  0  0  0.1  2
    """,
    reader="swc")

    assert np.isnan(
        neurite.volume_density(flat_neuron.neurites[0])
    )


def test_terminal_path_length_per_neurite():
    terminal_distances = [neurite.terminal_path_lengths(s) for s in SIMPLE.neurites]
    assert terminal_distances == [[10, 11], [10, 9]]


def test_max_radial_distance():
    assert_allclose([neurite.max_radial_distance(s) for s in SIMPLE.neurites],
                    [7.81025, 7.2111025])


def test_number_of_segments():
    assert [neurite.number_of_segments(s) for s in SIMPLE.neurites] == [3, 3]


def test_number_of_sections():
    assert [neurite.number_of_sections(s) for s in SIMPLE.neurites] == [3, 3]


def test_section_path_distances():
    path_lengths = [neurite.section_path_distances(s) for s in SIMPLE.neurites]
    assert path_lengths == [[5., 10., 11.], [4., 10., 9.]]


def test_section_term_lengths():
    term_lengths = [neurite.section_term_lengths(s) for s in SIMPLE.neurites]
    assert term_lengths == [[5., 6.], [6., 5.]]


def test_section_bif_lengths():
    bif_lengths = [neurite.section_bif_lengths(s) for s in SIMPLE.neurites]
    assert bif_lengths == [[5.],  [4.]]


def test_section_end_distances():
    end_dist = [neurite.section_end_distances(s) for s in SIMPLE.neurites]
    assert end_dist == [[5.0, 5.0, 6.0], [4.0, 6.0, 5.0]]


def test_section_partition_pairs():
    part_pairs = [neurite.partition_pairs(s) for s in SIMPLE.neurites]
    assert part_pairs == [[(1.0, 1.0)], [(1.0, 1.0)]]


def test_section_bif_radial_distances():
    bif_rads = [neurite.section_bif_radial_distances(s) for s in SIMPLE.neurites]
    assert bif_rads == [[5.],  [4.]]


def test_section_term_radial_distances():
    trm_rads = [neurite.section_term_radial_distances(s) for s in SIMPLE.neurites]
    assert_allclose(trm_rads, [[7.0710678118654755, 7.810249675906654], [7.211102550927978, 6.4031242374328485]])


def test_section_branch_orders():
    branch_orders = [neurite.section_branch_orders(s) for s in SIMPLE.neurites]
    assert_allclose(branch_orders, [[0, 1, 1], [0, 1, 1]])


def test_section_bif_branch_orders():
    bif_branch_orders = [neurite.section_bif_branch_orders(s) for s in SIMPLE.neurites]
    assert bif_branch_orders == [[0], [0]]


def test_section_term_branch_orders():
    term_branch_orders = [neurite.section_term_branch_orders(s) for s in SIMPLE.neurites]
    assert term_branch_orders == [[1, 1], [1, 1]]


def test_section_radial_distances():
    radial_distances = [neurite.section_radial_distances(s) for s in SIMPLE.neurites]
    assert_allclose(radial_distances,
                    [[5.0, sqrt(5**2 + 5**2), sqrt(6**2 + 5**2)],
                     [4.0, sqrt(6**2 + 4**2), sqrt(5**2 + 4**2)]])


def test_local_bifurcation_angles():
    local_bif_angles = [neurite.local_bifurcation_angles(s) for s in SIMPLE.neurites]
    assert_allclose(local_bif_angles, [[pi], [pi]])


def test_remote_bifurcation_angles():
    remote_bif_angles = [neurite.remote_bifurcation_angles(s) for s in SIMPLE.neurites]
    assert_allclose(remote_bif_angles, [[pi], [pi]])


def test_partition():
    partition = [neurite.bifurcation_partitions(s) for s in SIMPLE.neurites]
    assert_allclose(partition, [[1.0], [1.0]])


def test_partition_asymmetry():
    partition = [neurite.partition_asymmetry(s) for s in SIMPLE.neurites]
    assert_allclose(partition, [[0.0], [0.0]])

    partition = [neurite.partition_asymmetry(s, variant='length') for s in SIMPLE.neurites]
    assert_allclose(partition, [[0.0625], [0.06666666666666667]])

    with pytest.raises(ValueError):
        neurite.partition_asymmetry(SIMPLE, variant='invalid-variant')

    with pytest.raises(ValueError):
        neurite.partition_asymmetry(SIMPLE, method='invalid-method')


def test_segment_lengths():
    segment_lengths = [neurite.segment_lengths(s) for s in SIMPLE.neurites]
    assert_allclose(segment_lengths, [[5.0, 5.0, 6.0], [4.0, 6.0, 5.0]])


def test_segment_areas():
    result = [neurite.segment_areas(s) for s in SIMPLE.neurites]
    assert_allclose(result, [[31.415927, 16.019042, 19.109562], [25.132741, 19.109562, 16.019042]])


def test_segment_volumes():
    expected = [[15.70796327, 5.23598776, 6.28318531], [12.56637061, 6.28318531, 5.23598776]]
    result = [neurite.segment_volumes(s) for s in SIMPLE.neurites]
    assert_allclose(result, expected)


def test_segment_midpoints():
    midpoints = [neurite.segment_midpoints(s) for s in SIMPLE.neurites]
    assert_allclose(midpoints,
                    [[[0.,  (5. + 0) / 2,  0.],  # trunk type 2
                              [-2.5,  5.,  0.],
                              [3.,  5.,  0.]],
                              [[0., (-4. + 0) / 2.,  0.],  # trunk type 3
                              [3., -4.,  0.],
                              [-2.5, -4.,  0.]]])


def test_segment_radial_distances():
    """midpoints on segments."""
    radial_distances = [neurite.segment_radial_distances(s) for s in SIMPLE.neurites]
    assert_allclose(radial_distances,
                    [[2.5, sqrt(2.5**2 + 5**2), sqrt(3**2 + 5**2)], [2.0, 5.0, sqrt(2.5**2 + 4**2)]])


def test_segment_path_lengths():
    pathlengths = [neurite.segment_path_lengths(s) for s in SIMPLE.neurites]
    assert_allclose(pathlengths, [[5., 10., 11.], [4., 10., 9.]])

    pathlengths = neurite.segment_path_lengths(NRN.neurites[0])[:5]
    assert_allclose(pathlengths, [0.1, 1.332525, 2.5301487, 3.267878, 4.471462])


def test_section_taper_rates():
    assert_allclose(neurite.section_taper_rates(NRN.neurites[0])[:10],
                    [0.06776235492169848,
                     0.0588716599404923,
                     0.03791571485186163,
                     0.04674653812192691,
                     -0.026399800285566058,
                     -0.026547582897720887,
                     -0.045038414440432537,
                     0.02083822978267914,
                     -0.0027721371791201038,
                     0.0803069042861474],
                    atol=1e-4)
