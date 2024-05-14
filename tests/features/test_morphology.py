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

"""Test ``neurom.features.morphology``."""
from math import pi, sqrt
import tempfile
import warnings
from io import StringIO
from pathlib import Path

import morphio
import numpy as np
import pytest
from morphio import PointLevel, SectionType
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

from neurom import morphmath
from neurom import NeuriteType, load_morphology, AXON, BASAL_DENDRITE
from neurom.core import Morphology, Population
from neurom.exceptions import NeuroMError
from neurom.features import morphology, population, section


DATA_PATH = Path(__file__).parent.parent / 'data'
H5_PATH = DATA_PATH / 'h5/v1'
NRN = load_morphology(H5_PATH / 'Neuron.h5')
SWC_PATH = DATA_PATH / 'swc'
SIMPLE = load_morphology(SWC_PATH / 'simple.swc')
SIMPLE_TRUNK = load_morphology(SWC_PATH / 'simple_trunk.swc')
SWC_NRN = load_morphology(SWC_PATH / 'Neuron.swc')
with warnings.catch_warnings(record=True):
    SWC_NRN_3PT = load_morphology(SWC_PATH / 'soma' / 'three_pt_soma.swc')


def _add_neurite_trunk(morph, elevation, azimuth, neurite_type=SectionType.basal_dendrite):
    """Add a neurite from the elevation and azimuth to a given morphology."""
    mut = morph.to_morphio()
    if hasattr(mut, 'as_mutable'):
        mut = mut.as_mutable()

    new_pts = np.array(morphmath.vector_from_spherical(elevation, azimuth), ndmin=2)

    point_lvl = PointLevel(new_pts, [1])
    mut.append_root_section(point_lvl, neurite_type)
    return Morphology(mut)


def test_soma_volume():
    with warnings.catch_warnings(record=True):
        # SomaSinglePoint
        ret = morphology.soma_volume(SIMPLE)
        assert_almost_equal(ret, 4.1887902047863905)
        # SomaCylinders
        ret = morphology.soma_volume(SWC_NRN)
        assert_almost_equal(ret, 0.010726068245337955)
        # SomaSimpleContour
        ret = morphology.soma_volume(NRN)
        assert_almost_equal(ret, 0.0033147000251481135)
        # SomaNeuromorphoThreePointCylinders
        ret = morphology.soma_volume(SWC_NRN_3PT)
        assert_almost_equal(ret, 50.26548245743669)


def test_soma_surface_area():
    assert_allclose(morphology.soma_surface_area(SIMPLE), 12.566370614359172)
    assert_allclose(morphology.soma_surface_area(NRN), 0.1075095256160432)


def test_soma_radius():
    assert morphology.soma_radius(SIMPLE) == 1
    assert_allclose(morphology.soma_radius(NRN), 0.09249506049313666)


def test_total_area_per_neurite():
    def surface(r0, r1, h):
        return pi * (r0 + r1) * sqrt((r0 - r1) ** 2 + h**2)

    basal_area = surface(1, 1, 5) + surface(1, 0, 5) + surface(1, 0, 6)
    ret = morphology.total_area_per_neurite(SIMPLE, neurite_type=BASAL_DENDRITE)
    assert_almost_equal(ret[0], basal_area)

    axon_area = surface(1, 1, 4) + surface(1, 0, 5) + surface(1, 0, 6)
    ret = morphology.total_area_per_neurite(SIMPLE, neurite_type=AXON)
    assert_almost_equal(ret[0], axon_area)

    ret = morphology.total_area_per_neurite(SIMPLE)
    assert np.allclose(ret, [basal_area, axon_area])


def test_total_volume_per_neurite():
    vol = morphology.total_volume_per_neurite(NRN)
    assert len(vol) == 4

    # calculate the volumes by hand and compare
    vol2 = [sum(section.section_volume(s) for s in n.iter_sections()) for n in NRN.neurites]
    assert vol == vol2

    # regression test
    ref_vol = [271.94122143951864, 281.24754646913954, 274.98039928781355, 276.73860261723024]
    assert np.allclose(vol, ref_vol)


def test_total_length_per_neurite():
    total_lengths = morphology.total_length_per_neurite(SIMPLE)
    assert total_lengths == [5.0 + 5.0 + 6.0, 4.0 + 5.0 + 6.0]


def test_number_of_neurites():
    assert morphology.number_of_neurites(SIMPLE) == 2


def test_total_volume_per_neurite():
    # note: cannot use SIMPLE since it lies in a plane
    total_volumes = morphology.total_volume_per_neurite(NRN)
    assert_allclose(
        total_volumes,
        [271.94122143951864, 281.24754646913954, 274.98039928781355, 276.73860261723024],
    )


def test_number_of_sections_per_neurite():
    sections = morphology.number_of_sections_per_neurite(SIMPLE)
    assert_allclose(sections, (3, 3))


def test_trunk_section_lengths():
    ret = morphology.trunk_section_lengths(SIMPLE)
    assert ret == [5.0, 4.0]


def test_trunk_origin_radii():
    morph = load_swc(
        """
     1 1  0  0 0 1. -1
     2 3  0  0 0 1.0  1
     3 3  0  5 0 0.5  2
     4 3 -5  5 0 0.  3
     5 3  6  5 0 0.  3
     6 2  0  0 0 1.0  1
     7 2  0 -4 0 0.25  6
     8 2  6 -4 0 0.  7
     9 2 -5 -4 0 0.  7
    """
    )
    ret = morphology.trunk_origin_radii(morph)
    assert ret == [1.0, 1.0]

    ret = morphology.trunk_origin_radii(morph, min_length_filter=1.0)
    assert_array_almost_equal(ret, [0.5, 0.25])

    with pytest.warns(
        UserWarning,
        match=(
            r"In 'trunk_origin_radii': the 'min_length_filter' value is greater than the "
            r"path distance of the last point of the last section so the radius of this "
            r"point is returned\."
        ),
    ):
        ret = morphology.trunk_origin_radii(morph, min_length_filter=999)
    assert_array_almost_equal(ret, [0.5, 0.25])

    ret = morphology.trunk_origin_radii(morph, max_length_filter=15)
    assert_array_almost_equal(ret, [3.0 / 4, 5.0 / 8])

    ret = morphology.trunk_origin_radii(morph, min_length_filter=1, max_length_filter=15)
    assert_array_almost_equal(ret, [0.5, 0.25])

    with pytest.warns(
        UserWarning,
        match=(
            r"In 'trunk_origin_radii': the 'min_length_filter' and 'max_length_filter' "
            r"values excluded all the points of the section so the radius of the first "
            r"point after the 'min_length_filter' path distance is returned\."
        ),
    ):
        ret = morphology.trunk_origin_radii(morph, min_length_filter=0.1, max_length_filter=0.2)
    assert_array_almost_equal(ret, [0.5, 0.25])

    with pytest.raises(
        NeuroMError,
        match=(
            r"In 'trunk_origin_radii': the 'min_length_filter' value must be strictly greater "
            r"than 0\."
        ),
    ):
        ret = morphology.trunk_origin_radii(morph, min_length_filter=-999)

    with pytest.raises(
        NeuroMError,
        match=(
            r"In 'trunk_origin_radii': the 'max_length_filter' value must be strictly greater "
            r"than 0\."
        ),
    ):
        ret = morphology.trunk_origin_radii(morph, max_length_filter=-999)

    with pytest.raises(
        NeuroMError,
        match=(
            r"In 'trunk_origin_radii': the 'min_length_filter' value must be strictly less than the"
            r" 'max_length_filter' value\."
        ),
    ):
        ret = morphology.trunk_origin_radii(morph, min_length_filter=15, max_length_filter=5)


def test_trunk_origin_azimuths():
    ret = morphology.trunk_origin_azimuths(SIMPLE)
    assert ret == [0.0, 0.0]


def test_trunk_angles():
    ret = morphology.trunk_angles(SIMPLE_TRUNK)
    assert_array_almost_equal(ret, [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])
    ret = morphology.trunk_angles(SIMPLE_TRUNK, neurite_type=NeuriteType.basal_dendrite)
    assert_array_almost_equal(ret, [np.pi, np.pi])
    ret = morphology.trunk_angles(SIMPLE_TRUNK, neurite_type=NeuriteType.axon)
    assert_array_almost_equal(ret, [0.0])
    ret = morphology.trunk_angles(SIMPLE, neurite_type=NeuriteType.apical_dendrite)
    assert_array_almost_equal(ret, [])

    ret = morphology.trunk_angles(
        SIMPLE_TRUNK, coords_only=None, sort_along=None, consecutive_only=False
    )
    assert_array_almost_equal(
        ret,
        [
            [0.0, np.pi / 2, np.pi / 2, np.pi],
            [0.0, np.pi, np.pi / 2, np.pi / 2],
            [0.0, np.pi / 2, np.pi / 2, np.pi],
            [0.0, np.pi, np.pi / 2, np.pi / 2],
        ],
    )

    ret = morphology.trunk_angles(
        SIMPLE_TRUNK, coords_only="xyz", sort_along=None, consecutive_only=False
    )
    assert_array_almost_equal(
        ret,
        [
            [0.0, np.pi / 2, np.pi / 2, np.pi],
            [0.0, np.pi, np.pi / 2, np.pi / 2],
            [0.0, np.pi / 2, np.pi / 2, np.pi],
            [0.0, np.pi, np.pi / 2, np.pi / 2],
        ],
    )

    morph = load_morphology(SWC_PATH / 'simple_trunk.swc')

    # Add two basals
    morph = _add_neurite_trunk(morph, np.pi / 3, np.pi / 4)
    morph = _add_neurite_trunk(morph, -np.pi / 3, -np.pi / 4)

    ret = morphology.trunk_angles(morph)
    assert_array_almost_equal(ret, [np.pi / 2, 0.387596, 1.183199, 1.183199, 0.387596, np.pi / 2])
    ret = morphology.trunk_angles(morph, neurite_type=NeuriteType.basal_dendrite)
    assert_array_almost_equal(ret, [1.958393, 1.183199, 1.183199, 1.958393])
    ret = morphology.trunk_angles(morph, neurite_type=NeuriteType.axon)
    assert_array_almost_equal(ret, [0.0])
    ret = morphology.trunk_angles(morph, neurite_type=NeuriteType.apical_dendrite)
    assert_array_almost_equal(ret, [0.0])

    ret = morphology.trunk_angles(morph, coords_only=None, sort_along=None, consecutive_only=False)
    assert_array_almost_equal(
        ret,
        [
            [0.0, np.pi / 2, np.pi / 2, np.pi, 2.617993, np.pi / 6],
            [0.0, np.pi, np.pi / 2, 1.209429, 1.209429, np.pi / 2],
            [0.0, np.pi / 2, 1.932163, 1.932163, np.pi / 2, np.pi],
            [0.0, np.pi / 6, 2.617993, np.pi, np.pi / 2, np.pi / 2],
            [0.0, 2.418858, 2.617993, 1.209429, 1.932163, np.pi / 6],
            [0.0, np.pi / 6, 1.209429, 1.932163, 2.617993, 2.418858],
        ],
    )

    ret = morphology.trunk_angles(morph, coords_only="xyz", sort_along=None, consecutive_only=False)
    assert_array_almost_equal(
        ret,
        [
            [0.0, np.pi / 2, np.pi / 2, np.pi, 2.617993, np.pi / 6],
            [0.0, np.pi, np.pi / 2, 1.209429, 1.209429, np.pi / 2],
            [0.0, np.pi / 2, 1.932163, 1.932163, np.pi / 2, np.pi],
            [0.0, np.pi / 6, 2.617993, np.pi, np.pi / 2, np.pi / 2],
            [0.0, 2.418858, 2.617993, 1.209429, 1.932163, np.pi / 6],
            [0.0, np.pi / 6, 1.209429, 1.932163, 2.617993, 2.418858],
        ],
    )


def test_trunk_angles_inter_types():
    morph = load_morphology(SWC_PATH / 'simple_trunk.swc')

    # Add two basals
    morph = _add_neurite_trunk(morph, np.pi / 3, np.pi / 4)
    morph = _add_neurite_trunk(morph, -np.pi / 3, -np.pi / 4)

    # Test with no source
    ret = morphology.trunk_angles_inter_types(
        SIMPLE,
        NeuriteType.apical_dendrite,
        NeuriteType.basal_dendrite,
    )
    assert_array_almost_equal(ret, [])

    # Test default
    ret = morphology.trunk_angles_inter_types(
        morph,
        NeuriteType.apical_dendrite,
        NeuriteType.basal_dendrite,
        closest_component=None,
    )
    assert_array_almost_equal(
        ret,
        [
            [
                [np.pi / 2, -np.pi / 2, 0],
                [np.pi / 2, -np.pi / 2, np.pi],
                [np.pi / 6, -np.pi / 6, np.pi / 4],
                [5 * np.pi / 6, -5 * np.pi / 6, -np.pi / 4],
            ]
        ],
    )

    # Test with closest component equal to 3d angle
    ret = morphology.trunk_angles_inter_types(
        morph,
        NeuriteType.apical_dendrite,
        NeuriteType.basal_dendrite,
        closest_component=0,
    )
    assert_array_almost_equal(ret, [[[np.pi / 6, -np.pi / 6, np.pi / 4]]])

    # Test with only one target per source
    ret = morphology.trunk_angles_inter_types(
        morph,
        NeuriteType.basal_dendrite,
        NeuriteType.apical_dendrite,
        closest_component=None,
    )
    assert_array_almost_equal(
        ret,
        [
            [[np.pi / 2, np.pi / 2, 0]],
            [[np.pi / 2, np.pi / 2, -np.pi]],
            [[np.pi / 6, np.pi / 6, -np.pi / 4]],
            [[5 * np.pi / 6, 5 * np.pi / 6, np.pi / 4]],
        ],
    )

    # Test with only one target per source and closest component equal to 3d angle
    ret = morphology.trunk_angles_inter_types(
        morph,
        NeuriteType.basal_dendrite,
        NeuriteType.apical_dendrite,
        closest_component=0,
    )
    assert_array_almost_equal(
        ret,
        [
            [[np.pi / 2, np.pi / 2, 0]],
            [[np.pi / 2, np.pi / 2, -np.pi]],
            [[np.pi / 6, np.pi / 6, -np.pi / 4]],
            [[5 * np.pi / 6, 5 * np.pi / 6, np.pi / 4]],
        ],
    )


def test_trunk_angles_from_vector():
    morph = load_morphology(SWC_PATH / 'simple_trunk.swc')

    # Add two basals
    morph = _add_neurite_trunk(morph, np.pi / 3, np.pi / 4)
    morph = _add_neurite_trunk(morph, -np.pi / 3, -np.pi / 4)

    # Test with no neurite selected
    ret = morphology.trunk_angles_from_vector(
        SIMPLE,
        NeuriteType.apical_dendrite,
    )
    assert_array_almost_equal(ret, [])

    # Test default
    ret = morphology.trunk_angles_from_vector(
        morph,
        NeuriteType.basal_dendrite,
    )
    assert_array_almost_equal(
        ret,
        [
            [np.pi / 2, -np.pi / 2, 0],
            [np.pi / 2, -np.pi / 2, np.pi],
            [np.pi / 6, -np.pi / 6, np.pi / 4],
            [5 * np.pi / 6, -5 * np.pi / 6, -np.pi / 4],
        ],
    )

    # Test with given vector
    ret = morphology.trunk_angles_from_vector(morph, NeuriteType.basal_dendrite, vector=(0, -1, 0))
    assert_array_almost_equal(
        ret,
        [
            [np.pi / 2, np.pi / 2, 0],
            [np.pi / 2, np.pi / 2, np.pi],
            [5 * np.pi / 6, 5 * np.pi / 6, np.pi / 4],
            [np.pi / 6, np.pi / 6, -np.pi / 4],
        ],
    )


def test_trunk_vectors():
    ret = morphology.trunk_vectors(SIMPLE_TRUNK)
    assert_array_equal(ret[0], [0.0, -1.0, 0.0])
    assert_array_equal(ret[1], [1.0, 0.0, 0.0])
    assert_array_equal(ret[2], [-1.0, 0.0, 0.0])
    assert_array_equal(ret[3], [0.0, 1.0, 0.0])
    ret = morphology.trunk_vectors(SIMPLE_TRUNK, neurite_type=NeuriteType.axon)
    assert_array_equal(ret[0], [0.0, -1.0, 0.0])


def test_trunk_origin_elevations():
    n0 = load_morphology(
        StringIO(
            u"""
    1 1 0 0 0 4 -1
    2 3 1 0 0 2 1
    3 3 2 1 1 2 2
    4 3 0 1 0 2 1
    5 3 1 2 1 2 4
    """
        ),
        reader='swc',
    )

    n1 = load_morphology(
        StringIO(
            u"""
    1 1 0 0 0 4 -1
    2 3 0 -1 0 2 1
    3 3 -1 -2 -1 2 2
    """
        ),
        reader='swc',
    )

    pop = [n0, n1]
    assert_allclose(morphology.trunk_origin_elevations(n0), [0.0, np.pi / 2.0])
    assert_allclose(morphology.trunk_origin_elevations(n1), [-np.pi / 2.0])
    assert_allclose(
        morphology.trunk_origin_elevations(n0, NeuriteType.basal_dendrite), [0.0, np.pi / 2.0]
    )
    assert_allclose(
        morphology.trunk_origin_elevations(n1, NeuriteType.basal_dendrite), [-np.pi / 2.0]
    )

    assert morphology.trunk_origin_elevations(n0, NeuriteType.axon) == []
    assert morphology.trunk_origin_elevations(n1, NeuriteType.axon) == []
    assert morphology.trunk_origin_elevations(n0, NeuriteType.apical_dendrite) == []
    assert morphology.trunk_origin_elevations(n1, NeuriteType.apical_dendrite) == []


def test_trunk_elevation_zero_norm_vector_raises():
    with pytest.raises(Exception):
        morphology.trunk_origin_elevations(SWC_NRN)


def test_sholl_crossings_simple(SIMPLE_MORPHOLOGY):
    center = SIMPLE_MORPHOLOGY.soma.center
    radii = []
    assert list(morphology.sholl_crossings(SIMPLE_MORPHOLOGY, center=center, radii=radii)) == []
    assert list(morphology.sholl_crossings(SIMPLE_MORPHOLOGY, radii=radii)) == []
    assert list(morphology.sholl_crossings(SIMPLE_MORPHOLOGY)) == [2]

    radii = [1.0]
    assert list(morphology.sholl_crossings(SIMPLE_MORPHOLOGY, center=center, radii=radii)) == [2]

    radii = [1.0, 5.1]
    assert list(morphology.sholl_crossings(SIMPLE_MORPHOLOGY, center=center, radii=radii)) == [2, 4]

    radii = [1.0, 4.0, 5.0]
    assert list(morphology.sholl_crossings(SIMPLE_MORPHOLOGY, center=center, radii=radii)) == [
        2,
        4,
        5,
    ]

    assert list(
        morphology.sholl_crossings(SIMPLE_MORPHOLOGY.sections[:2], center=center, radii=radii)
    ) == [1, 1, 2]

    radii = [1.0, 4.0, 5.0, 10]
    assert list(
        morphology.sholl_crossings(
            SIMPLE_MORPHOLOGY, neurite_type=NeuriteType.all, center=center, radii=radii
        )
    ) == [2, 4, 5, 0]
    assert list(
        morphology.sholl_crossings(
            SIMPLE_MORPHOLOGY, neurite_type=NeuriteType.basal_dendrite, center=center, radii=radii
        )
    ) == [1, 1, 3, 0]
    assert list(
        morphology.sholl_crossings(
            SIMPLE_MORPHOLOGY, neurite_type=NeuriteType.apical_dendrite, center=center, radii=radii
        )
    ) == [0, 0, 0, 0]
    assert list(
        morphology.sholl_crossings(
            SIMPLE_MORPHOLOGY, neurite_type=NeuriteType.axon, center=center, radii=radii
        )
    ) == [1, 3, 2, 0]


def test_sholl_frequency_simple(SIMPLE_MORPHOLOGY):
    assert list(morphology.sholl_frequency(SIMPLE_MORPHOLOGY)) == [2]
    assert list(morphology.sholl_frequency(SIMPLE_MORPHOLOGY, step_size=3)) == [2, 4, 3]
    assert list(morphology.sholl_frequency(SIMPLE_MORPHOLOGY, bins=[1, 3, 5])) == [2, 2, 5]

    assert list(
        morphology.sholl_frequency(SIMPLE_MORPHOLOGY, neurite_type=NeuriteType.basal_dendrite)
    ) == [1]
    assert list(
        morphology.sholl_frequency(
            SIMPLE_MORPHOLOGY, neurite_type=NeuriteType.basal_dendrite, step_size=3
        )
    ) == [1, 1, 2]
    assert list(
        morphology.sholl_frequency(
            SIMPLE_MORPHOLOGY, neurite_type=NeuriteType.basal_dendrite, bins=[1, 3, 5]
        )
    ) == [1, 1, 3]

    assert list(morphology.sholl_frequency(SIMPLE_MORPHOLOGY, neurite_type=NeuriteType.axon)) == [1]
    assert list(
        morphology.sholl_frequency(SIMPLE_MORPHOLOGY, neurite_type=NeuriteType.axon, step_size=3)
    ) == [1, 3, 1]
    assert list(
        morphology.sholl_frequency(SIMPLE_MORPHOLOGY, neurite_type=NeuriteType.axon, bins=[1, 3, 5])
    ) == [1, 1, 2]


def load_swc(string):
    with tempfile.NamedTemporaryFile(prefix='test_morphology', mode='w', suffix='.swc') as fd:
        fd.write(string)
        fd.flush()
        return load_morphology(fd.name)


def test_sholl_analysis_custom():
    # recreate morphs from Fig 2 of
    # http://dx.doi.org/10.1016/j.jneumeth.2014.01.016
    radii = np.arange(10, 81, 10)
    center = 0, 0, 0
    morph_A = load_swc(
        """\
 1 1   0  0  0 1. -1
 2 3   0  0  0 1.  1
 3 3  80  0  0 1.  2
 4 4   0  0  0 1.  1
 5 4 -80  0  0 1.  4"""
    )
    assert list(morphology.sholl_crossings(morph_A, center=center, radii=radii)) == [
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    ]

    morph_B = load_swc(
        """\
 1 1   0   0  0 1. -1
 2 3   0   0  0 1.  1
 3 3  35   0  0 1.  2
 4 3  51  10  0 1.  3
 5 3  51   5  0 1.  3
 6 3  51   0  0 1.  3
 7 3  51  -5  0 1.  3
 8 3  51 -10  0 1.  3
 9 4 -35   0  0 1.  2
10 4 -51  10  0 1.  9
11 4 -51   5  0 1.  9
12 4 -51   0  0 1.  9
13 4 -51  -5  0 1.  9
14 4 -51 -10  0 1.  9
                       """
    )
    assert list(morphology.sholl_crossings(morph_B, center=center, radii=radii)) == [
        2,
        2,
        2,
        10,
        10,
        0,
        0,
        0,
    ]

    morph_C = load_swc(
        """\
 1 1   0   0  0 1. -1
 2 3   0   0  0 1.  1
 3 3  65   0  0 1.  2
 4 3  85  10  0 1.  3
 5 3  85   5  0 1.  3
 6 3  85   0  0 1.  3
 7 3  85  -5  0 1.  3
 8 3  85 -10  0 1.  3
 9 4  65   0  0 1.  2
10 4  85  10  0 1.  9
11 4  85   5  0 1.  9
12 4  85   0  0 1.  9
13 4  85  -5  0 1.  9
14 4  85 -10  0 1.  9
                       """
    )
    assert list(morphology.sholl_crossings(morph_C, center=center, radii=radii)) == [
        2,
        2,
        2,
        2,
        2,
        2,
        10,
        10,
    ]


def test_extent_along_axis():
    morph = load_swc(
        """
        1 1   0  0   0 1. -1
        2 3   0  -60 0 1.  1
        3 3  80  0   2 1.  2
        4 4   0  60  3 1.  1
        5 4 -80  0.  0 1.  4
    """
    )
    assert_almost_equal(morphology._extent_along_axis(morph, 0, NeuriteType.all), 160.0)
    assert_almost_equal(morphology._extent_along_axis(morph, 1, NeuriteType.all), 120.0)
    assert_almost_equal(morphology._extent_along_axis(morph, 2, NeuriteType.all), 3.0)


def test_total_width():
    morph = load_swc(
        """
        1 1   0  0   0 1. -1
        2 3   0  -60 0 1.  1
        3 3  80  0   2 1.  2
        4 4   0  60  3 1.  1
        5 4 -80  0.  0 1.  4
    """
    )
    assert_almost_equal(morphology.total_width(morph, neurite_type=NeuriteType.axon), 0.0)
    assert_almost_equal(
        morphology.total_width(morph, neurite_type=NeuriteType.basal_dendrite), 80.0
    )
    assert_almost_equal(
        morphology.total_width(morph, neurite_type=NeuriteType.apical_dendrite), 80.0
    )


def test_total_height():
    morph = load_swc(
        """
        1 1   0  0   0 1. -1
        2 3   0  -60 0 1.  1
        3 3  80  0   2 1.  2
        4 4   0  60  3 1.  1
        5 4 -80  0.  0 1.  4
    """
    )
    assert_almost_equal(morphology.total_height(morph, neurite_type=NeuriteType.axon), 0.0)
    assert_almost_equal(
        morphology.total_height(morph, neurite_type=NeuriteType.basal_dendrite), 60.0
    )
    assert_almost_equal(
        morphology.total_height(morph, neurite_type=NeuriteType.apical_dendrite), 60.0
    )


def test_total_depth():
    morph = load_swc(
        """
        1 1   0  0   0 1. -1
        2 3   0  -60 0 1.  1
        3 3  80  0   2 1.  2
        4 4   0  60  3 1.  1
        5 4 -80  0.  0 1.  4
    """
    )
    assert_almost_equal(morphology.total_depth(morph, neurite_type=NeuriteType.axon), 0.0)
    assert_almost_equal(morphology.total_depth(morph, neurite_type=NeuriteType.basal_dendrite), 2.0)
    assert_almost_equal(
        morphology.total_depth(morph, neurite_type=NeuriteType.apical_dendrite), 3.0
    )


def test_volume_density():
    morph = load_swc(
        """
        1  1   0.5      0.5      0.5        0.5 -1
        2  3   0.211324 0.211324 0.788675   0.1  1
        3  3   0.0      0.0      1.0        0.1  2
        4  3   0.211324 0.788675 0.788675   0.1  1
        5  3   0.0      1.0      1.0        0.1  4
        6  3   0.788675 0.211324 0.788675   0.1  1
        7  3   1.0      0.0      1.0        0.1  6
        8  3   0.211324 0.211324 0.211324   0.1  1
        9  3   0.0      0.0      0.0        0.1  8
       10  3   0.211324 0.788675 0.211324   0.1  1
       11  3   0.0      1.0      0.0        0.1  10
       12  5   0.788675 0.788675 0.211324   0.1  1
       13  5   1.0      1.0      0.0        0.1  12
       14  2   0.788675 0.211324 0.211324   0.1  1
       15  2   1.0      0.0      0.0        0.1  14
       16  3   0.788675 0.788675 0.788675   0.1  1
       17  3   1.0      1.0      1.0        0.1  16
    """
    )

    # the neurites sprout from the center of a cube to its vertices, therefore the convex hull
    # is the cube itself of side 1.0
    expected_hull_volume = 1.0

    # diagonal - radius
    expected_neurite_length = np.sqrt(3) * 0.5 - 0.5

    # distance from center of unit cube to its vertices is sqrt(3)
    expected_neurite_volume = np.pi * 0.1**2 * expected_neurite_length * 8

    expected_volume_density = expected_neurite_volume / expected_hull_volume

    assert_almost_equal(morphology.volume_density(morph), expected_volume_density, decimal=5)
    assert_almost_equal(
        morphology.volume_density(morph, neurite_type=NeuriteType.all),
        expected_volume_density,
        decimal=5,
    )

    # (0 0 1) (0 1 1) (0 0 0) (0 1 0) (1 0 1)(1 1 1)
    # form a triangular prism
    # Volume = triangle_area * depth = 0.5 * 1. * 1. * 1.
    expected_hull_volume = 0.5

    expected_neurite_volume = np.pi * 0.1**2 * expected_neurite_length * 6

    expected_volume_density = expected_neurite_volume / expected_hull_volume

    assert_almost_equal(
        morphology.volume_density(morph, neurite_type=NeuriteType.basal_dendrite),
        expected_volume_density,
        decimal=5,
    )

    # invalid convex hull
    assert np.isnan(
        morphology.volume_density(morph, neurite_type=NeuriteType.axon),
    )

    # no points
    assert np.isnan(
        morphology.volume_density(morph, neurite_type=NeuriteType.apical_dendrite),
    )


def test_unique_projected_points():
    morph = load_swc(
        """
        1  1   0.5 0.5 0.5   0.5 -1
        2  3   0.2 0.2 0.7   0.1  1
        3  3   0.0 0.0 1.0   0.1  2
        4  3   0.2 0.7 0.7   0.1  1
        5  3   0.0 1.0 1.0   0.1  4
        6  3   0.7 0.2 0.7   0.1  1
        7  3   1.0 0.0 1.0   0.1  6
        8  3   0.2 0.2 0.2   0.1  1
        9  3   0.0 0.0 0.0   0.1  8
       10  3   0.2 0.7 0.2   0.1  1
       11  3   0.0 1.0 0.0   0.1  10
       12  5   0.7 0.7 0.2   0.1  1
       13  5   1.0 1.0 0.0   0.1  12
       14  2   0.7 0.2 0.2   0.1  1
       15  2   1.0 0.0 0.0   0.1  14
       16  3   0.7 0.7 0.7   0.1  1
       17  3   1.0 1.0 1.0   0.1  16
    """
    )

    for plane, enalp in zip(("xy", "xz", "yz"), ("yx", "zx", "zy")):
        assert_allclose(
            morphology._unique_projected_points(morph, plane, NeuriteType.all),
            morphology._unique_projected_points(morph, enalp, NeuriteType.all),
        )

    assert_allclose(
        morphology._unique_projected_points(morph, "xy", NeuriteType.all),
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.2, 0.2],
            [0.2, 0.2],
            [0.2, 0.7],
            [0.2, 0.7],
            [0.7, 0.2],
            [0.7, 0.2],
            [0.7, 0.7],
            [0.7, 0.7],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
    )
    assert_allclose(
        morphology._unique_projected_points(morph, "xz", NeuriteType.all),
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.2, 0.2],
            [0.2, 0.7],
            [0.2, 0.2],
            [0.2, 0.7],
            [0.7, 0.2],
            [0.7, 0.7],
            [0.7, 0.2],
            [0.7, 0.7],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
    )
    assert_allclose(
        morphology._unique_projected_points(morph, "yz", NeuriteType.all),
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.2],
            [0.2, 0.7],
            [0.7, 0.2],
            [0.7, 0.7],
            [0.2, 0.2],
            [0.2, 0.7],
            [0.7, 0.2],
            [0.7, 0.7],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
    )

    with pytest.raises(NeuroMError):
        morphology._unique_projected_points(morph, "airplane", NeuriteType.all)

    assert len(morphology._unique_projected_points(morph, "yz", NeuriteType.apical_dendrite)) == 0


def test_missing_soma():
    NRN_missing_soma = load_morphology(SWC_PATH / 'Single_apical_no_soma.swc')

    with pytest.raises(NeuroMError):
        morphology.trunk_origin_elevations(NRN_missing_soma)
    with pytest.raises(NeuroMError):
        morphology.trunk_origin_azimuths(NRN_missing_soma)
    with pytest.raises(NeuroMError):
        morphology.trunk_origin_elevations(NRN_missing_soma)
    with pytest.raises(NeuroMError):
        morphology.trunk_vectors(NRN_missing_soma)
    with pytest.raises(NeuroMError):
        morphology.sholl_crossings(NRN_missing_soma)
    with pytest.raises(NeuroMError):
        morphology.sholl_frequency(NRN_missing_soma)
    with pytest.raises(NeuroMError):
        morphology.length_fraction_above_soma(NRN_missing_soma)

    POP_missing_soma = Population([NRN_missing_soma])
    with pytest.raises(NeuroMError):
        population.sholl_frequency(POP_missing_soma)
