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

"""Test neurom._neuritefunc functionality."""

from math import pi, sqrt
from pathlib import Path

import neurom as nm
import numpy as np
import scipy
from mock import patch
from neurom.features import neuritefunc as _nf
from neurom.features import sectionfunc as sectionfunc
from neurom.geom import convex_hull

import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_almost_equal
from utils import _close

DATA_PATH = Path(__file__).parent.parent / 'data'
H5_PATH = DATA_PATH / 'h5/v1'
SWC_PATH = DATA_PATH / 'swc'
SIMPLE = nm.load_neuron(SWC_PATH / 'simple.swc')
NRN = nm.load_neuron(H5_PATH / 'Neuron.h5')


def test_principal_direction_extents():
    principal_dir = list(_nf.principal_direction_extents(SIMPLE))
    assert_allclose(principal_dir,
                    (14.736052694538641, 12.105102672688004))

    # test with a realistic neuron
    nrn = nm.load_neuron(Path(H5_PATH, 'bio_neuron-000.h5'))

    p_ref = [1672.9694359427331, 142.43704397865031, 226.45895382204986,
             415.50612748523838, 429.83008974193206, 165.95410536922873,
             346.83281498399697]

    p = _nf.principal_direction_extents(nrn)
    _close(np.array(p), np.array(p_ref))


def test_n_bifurcation_points():
    assert _nf.n_bifurcation_points(SIMPLE.neurites[0]) == 1
    assert _nf.n_bifurcation_points(SIMPLE.neurites[1]) == 1
    assert _nf.n_bifurcation_points(SIMPLE.neurites) == 2


def test_n_forking_points():
    assert _nf.n_forking_points(SIMPLE.neurites[0]) == 1
    assert _nf.n_forking_points(SIMPLE.neurites[1]) == 1
    assert _nf.n_forking_points(SIMPLE.neurites) == 2


def test_n_leaves():
    assert _nf.n_leaves(SIMPLE.neurites[0]) == 2
    assert _nf.n_leaves(SIMPLE.neurites[1]) == 2
    assert _nf.n_leaves(SIMPLE.neurites) == 4


def test_total_area_per_neurite():
    def surface(r0, r1, h):
        return pi * (r0 + r1) * sqrt((r0 - r1) ** 2 + h ** 2)

    basal_area = surface(1, 1, 5) + surface(1, 0, 5) + surface(1, 0, 6)
    ret = _nf.total_area_per_neurite(SIMPLE,
                                     neurite_type=nm.BASAL_DENDRITE)
    assert_almost_equal(ret[0], basal_area)

    axon_area = surface(1, 1, 4) + surface(1, 0, 5) + surface(1, 0, 6)
    ret = _nf.total_area_per_neurite(SIMPLE, neurite_type=nm.AXON)
    assert_almost_equal(ret[0], axon_area)

    ret = _nf.total_area_per_neurite(SIMPLE)
    assert np.allclose(ret, [basal_area, axon_area])


def test_total_volume_per_neurite():
    vol = _nf.total_volume_per_neurite(NRN)
    assert len(vol) == 4

    # calculate the volumes by hand and compare
    vol2 = [sum(sectionfunc.section_volume(s) for s in n.iter_sections())
            for n in NRN.neurites
            ]
    assert vol == vol2

    # regression test
    ref_vol = [271.94122143951864, 281.24754646913954,
               274.98039928781355, 276.73860261723024]
    assert np.allclose(vol, ref_vol)


def test_neurite_volume_density():

    vol = np.array(_nf.total_volume_per_neurite(NRN))
    hull_vol = np.array([convex_hull(n).volume for n in nm.iter_neurites(NRN)])

    vol_density = _nf.neurite_volume_density(NRN)
    assert len(vol_density) == 4
    assert np.allclose(vol_density, vol / hull_vol)

    ref_density = [0.43756606998299519, 0.52464681266899216,
                   0.24068543213643726, 0.26289304906104355]
    assert_allclose(vol_density, ref_density)


def test_neurite_volume_density_failed_convex_hull():
    with patch('neurom.features.neuritefunc.convex_hull',
               side_effect=scipy.spatial.qhull.QhullError('boom')):
        vol_density = _nf.neurite_volume_density(NRN)
        assert vol_density, np.nan


def test_terminal_path_length_per_neurite():
    terminal_distances = _nf.terminal_path_lengths_per_neurite(SIMPLE)
    assert_allclose(terminal_distances,
                    (5 + 5., 5 + 6., 4. + 6., 4. + 5))
    terminal_distances = _nf.terminal_path_lengths_per_neurite(SIMPLE,
                                                               neurite_type=nm.AXON)
    assert_allclose(terminal_distances,
                    (4. + 6., 4. + 5.))


def test_total_length_per_neurite():
    total_lengths = _nf.total_length_per_neurite(SIMPLE)
    assert_allclose(total_lengths,
                    (5. + 5. + 6., 4. + 5. + 6.))


def test_max_radial_distance():
    dmax = _nf.max_radial_distance(SIMPLE)
    assert_almost_equal(dmax, 7.81025, decimal=6)


def test_n_segments():
    n_segments = _nf.n_segments(SIMPLE)
    assert n_segments == 6


def test_n_neurites():
    n_neurites = _nf.n_neurites(SIMPLE)
    assert n_neurites == 2


def test_n_sections():
    n_sections = _nf.n_sections(SIMPLE)
    assert n_sections == 6


def test_neurite_volumes():
    # note: cannot use SIMPLE since it lies in a plane
    total_volumes = _nf.total_volume_per_neurite(NRN)
    assert_allclose(total_volumes,
                    [271.94122143951864, 281.24754646913954,
                     274.98039928781355, 276.73860261723024]
                    )


def test_section_path_lengths():
    path_lengths = list(_nf.section_path_lengths(SIMPLE))
    assert_allclose(path_lengths,
                    (5., 10., 11.,  # type 3, basal dendrite
                     4., 10., 9.))  # type 2, axon


def test_section_term_lengths():
    term_lengths = list(_nf.section_term_lengths(SIMPLE))
    assert_allclose(term_lengths,
                    (5., 6., 6., 5.))


def test_section_bif_lengths():
    bif_lengths = list(_nf.section_bif_lengths(SIMPLE))
    assert_allclose(bif_lengths,
                    (5.,  4.))


def test_section_end_distances():
    end_dist = list(_nf.section_end_distances(SIMPLE))
    assert_allclose(end_dist,
                    [5.0, 5.0, 6.0, 4.0, 6.0, 5.0])


def test_section_partition_pairs():
    part_pairs = list(_nf.partition_pairs(SIMPLE))
    assert_allclose(part_pairs,
                    [(1.0, 1.0), (1.0, 1.0)])


def test_section_bif_radial_distances():
    bif_rads = list(_nf.section_bif_radial_distances(SIMPLE))
    assert_allclose(bif_rads,
                    [5.,  4.])
    trm_rads = list(_nf.section_bif_radial_distances(NRN, neurite_type=nm.AXON))
    assert_allclose(trm_rads,
                    [8.842008561870646,
                     16.7440421479104,
                     23.070306480850533,
                     30.181121708042546,
                     36.62766031035137,
                     43.967487830324885,
                     51.91971040624528,
                     59.427722328770955,
                     66.25222507299583,
                     74.05119754074926])


def test_section_term_radial_distances():
    trm_rads = list(_nf.section_term_radial_distances(SIMPLE))
    assert_allclose(trm_rads,
                    [7.0710678118654755, 7.810249675906654, 7.211102550927978, 6.4031242374328485])
    trm_rads = list(_nf.section_term_radial_distances(NRN, neurite_type=nm.APICAL_DENDRITE))
    assert_allclose(trm_rads,
                    [16.22099879395879,
                     25.992977561564082,
                     33.31600613822663,
                     42.721314797308175,
                     52.379508081911546,
                     59.44327819128149,
                     67.07832724133213,
                     79.97743930553612,
                     87.10434825508366,
                     97.25246040544428,
                     99.58945832481642])


def test_number_of_sections_per_neurite():
    sections = _nf.number_of_sections_per_neurite(SIMPLE)
    assert_allclose(sections,
                    (3, 3))


def test_section_branch_orders():
    branch_orders = list(_nf.section_branch_orders(SIMPLE))
    assert_allclose(branch_orders,
                    (0, 1, 1,  # type 3, basal dendrite
                     0, 1, 1))  # type 2, axon


def test_section_bif_branch_orders():
    bif_branch_orders = list(_nf.section_bif_branch_orders(SIMPLE))
    assert_allclose(bif_branch_orders,
                    (0,  # type 3, basal dendrite
                     0))  # type 2, axon


def test_section_term_branch_orders():
    term_branch_orders = list(_nf.section_term_branch_orders(SIMPLE))
    assert_allclose(term_branch_orders,
                    (1, 1,  # type 3, basal dendrite
                     1, 1))  # type 2, axon


def test_section_radial_distances():
    radial_distances = _nf.section_radial_distances(SIMPLE)
    assert_allclose(radial_distances,
                    (5.0, sqrt(5**2 + 5**2), sqrt(6**2 + 5**2),   # type 3, basal dendrite
                     4.0, sqrt(6**2 + 4**2), sqrt(5**2 + 4**2)))  # type 2, axon


def test_local_bifurcation_angles():
    local_bif_angles = list(_nf.local_bifurcation_angles(SIMPLE))
    assert_allclose(local_bif_angles,
                    (pi, pi))


def test_remote_bifurcation_angles():
    remote_bif_angles = list(_nf.remote_bifurcation_angles(SIMPLE))
    assert_allclose(remote_bif_angles,
                    (pi, pi))


def test_partition():
    partition = list(_nf.bifurcation_partitions(SIMPLE))
    assert_allclose(partition,
                    (1.0, 1.0))


def test_partition_asymmetry():
    partition = list(_nf.partition_asymmetries(SIMPLE))
    assert_allclose(partition,
                    (0.0, 0.0))

    partition = list(_nf.partition_asymmetries(SIMPLE, variant='length'))
    assert_allclose(partition,
                    (0.0625, 0.06666666666666667))

    with pytest.raises(ValueError):
        _nf.partition_asymmetries(SIMPLE, variant='unvalid-variant')


def test_segment_lengths():
    segment_lengths = _nf.segment_lengths(SIMPLE)
    assert_allclose(segment_lengths,
                    (5.0, 5.0, 6.0,   # type 3, basal dendrite
                     4.0, 6.0, 5.0))  # type 2, axon


def test_segment_areas():
    result = _nf.segment_areas(SIMPLE)
    assert_allclose(result,
                    [31.415927,
                     16.019042,
                     19.109562,
                     25.132741,
                     19.109562,
                     16.019042])


def test_segment_volumes():
    expected = [
        15.70796327,
        5.23598776,
        6.28318531,
        12.56637061,
        6.28318531,
        5.23598776,
    ]
    result = _nf.segment_volumes(SIMPLE)
    assert_allclose(result, expected)


def test_segment_midpoints():
    midpoints = np.array(_nf.segment_midpoints(SIMPLE))
    assert_allclose(midpoints,
                    np.array([[0.,  (5. + 0) / 2,  0.],  # trunk type 2
                              [-2.5,  5.,  0.],
                              [3.,  5.,  0.],
                              [0., (-4. + 0) / 2.,  0.],  # trunk type 3
                              [3., -4.,  0.],
                              [-2.5, -4.,  0.]]))


def test_segment_radial_distances():
    """midpoints on segments."""
    radial_distances = _nf.segment_radial_distances(SIMPLE)
    assert_allclose(radial_distances,
                    [2.5, sqrt(2.5**2 + 5**2), sqrt(3**2 + 5**2), 2.0, 5.0, sqrt(2.5**2 + 4**2)])


def test_segment_path_lengths():
    pathlengths = _nf.segment_path_lengths(SIMPLE)
    assert_allclose(pathlengths, [5., 10., 11., 4., 10., 9.])

    pathlengths = _nf.segment_path_lengths(NRN)[:5]
    assert_array_almost_equal(pathlengths, [0.1, 1.332525, 2.530149, 3.267878, 4.471462])


def test_principal_direction_extents():
    principal_dir = list(_nf.principal_direction_extents(SIMPLE))
    assert_allclose(principal_dir,
                    (14.736052694538641, 12.105102672688004))


def test_section_taper_rates():
    assert_allclose(list(_nf.section_taper_rates(NRN.neurites[0]))[:10],
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
