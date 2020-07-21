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

"""Test neurom._neuronfunc functionality."""
from pathlib import Path
import tempfile
import warnings
from io import StringIO

import numpy as np
from nose import tools as nt
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal)

from neurom import NeuriteType, load_neuron
from neurom.core.population import Population

# NOTE: The 'bf' alias is used in the fst/tests modules
# Do NOT change it.
# TODO: If other neurom.features are imported,
# the should use the aliasing used in fst/tests module files
from neurom.features import neuronfunc as _nf

DATA_PATH = Path(__file__).parent.parent.parent.parent / 'test_data'
H5_PATH = DATA_PATH / 'h5/v1'
NRN = load_neuron(Path(H5_PATH, 'Neuron.h5'))

SWC_PATH = DATA_PATH / 'swc'
SIMPLE = load_neuron(Path(SWC_PATH, 'simple.swc'))
SIMPLE_TRUNK = load_neuron(Path(SWC_PATH, 'simple_trunk.swc'))
SWC_NRN = load_neuron(Path(SWC_PATH, 'Neuron.swc'))
with warnings.catch_warnings(record=True):
    SWC_NRN_3PT = load_neuron(Path(SWC_PATH, 'soma', 'three_pt_soma.swc'))


def test_soma_volume():
    with warnings.catch_warnings(record=True):
        # SomaSinglePoint
        ret = _nf.soma_volume(SIMPLE)
        assert_almost_equal(ret, 4.1887902047863905)
        # SomaCylinders
        ret = _nf.soma_volume(SWC_NRN)
        assert_almost_equal(ret, 0.010726068245337955)
        # SomaSimpleContour
        ret = _nf.soma_volume(NRN)
        assert_almost_equal(ret, 0.0033147000251481135)
        # SomaNeuromorphoThreePointCylinders
        ret = _nf.soma_volume(SWC_NRN_3PT)
        assert_almost_equal(ret, 50.26548245743669)


def test_soma_volumes():
    with warnings.catch_warnings(record=True):
        ret = _nf.soma_volumes(SIMPLE)
        nt.eq_(ret, [4.1887902047863905, ])


def test_soma_surface_area():
    ret = _nf.soma_surface_area(SIMPLE)
    nt.eq_(ret, 12.566370614359172)


def test_soma_surface_areas():
    ret = _nf.soma_surface_areas(SIMPLE)
    nt.eq_(ret, [12.566370614359172, ])


def test_soma_radii():
    ret = _nf.soma_radii(SIMPLE)
    nt.eq_(ret, [1., ])


def test_trunk_section_lengths():
    ret = _nf.trunk_section_lengths(SIMPLE)
    nt.eq_(ret, [5.0, 4.0])


def test_trunk_origin_radii():
    ret = _nf.trunk_origin_radii(SIMPLE)
    nt.eq_(ret, [1.0, 1.0])


def test_trunk_origin_azimuths():
    ret = _nf.trunk_origin_azimuths(SIMPLE)
    nt.eq_(ret, [0.0, 0.0])


def test_trunk_angles():
    ret = _nf.trunk_angles(SIMPLE_TRUNK)
    assert_array_almost_equal(ret, [np.pi/2, np.pi/2, np.pi/2, np.pi/2])
    ret = _nf.trunk_angles(SIMPLE_TRUNK, neurite_type=NeuriteType.basal_dendrite)
    assert_array_almost_equal(ret, [np.pi, np.pi])
    ret = _nf.trunk_angles(SIMPLE_TRUNK, neurite_type=NeuriteType.axon)
    assert_array_almost_equal(ret, [0.0])
    ret = _nf.trunk_angles(SIMPLE, neurite_type=NeuriteType.apical_dendrite)
    assert_array_almost_equal(ret, [])


def test_trunk_vectors():
    ret = _nf.trunk_vectors(SIMPLE_TRUNK)
    assert_array_equal(ret[0], [0., -1.,  0.])
    assert_array_equal(ret[1], [1.,  0.,  0.])
    assert_array_equal(ret[2], [-1.,  0.,  0.])
    assert_array_equal(ret[3], [0.,  1.,  0.])
    ret = _nf.trunk_vectors(SIMPLE_TRUNK, neurite_type=NeuriteType.axon)
    assert_array_equal(ret[0], [0., -1.,  0.])


def test_trunk_origin_elevations():
    n0 = load_neuron(StringIO(u"""
    1 1 0 0 0 4 -1
    2 3 1 0 0 2 1
    3 3 2 1 1 2 2
    4 3 0 1 0 2 1
    5 3 1 2 1 2 4
    """), reader='swc')

    n1 = load_neuron(StringIO(u"""
    1 1 0 0 0 4 -1
    2 3 0 -1 0 2 1
    3 3 -1 -2 -1 2 2
    """), reader='swc')

    pop = Population([n0, n1])
    assert_array_equal(_nf.trunk_origin_elevations(pop),
                       [0.0, np.pi/2., -np.pi/2.])

    assert_array_equal(_nf.trunk_origin_elevations(pop, neurite_type=NeuriteType.basal_dendrite),
                       [0.0, np.pi/2., -np.pi/2.])

    assert_array_equal(_nf.trunk_origin_elevations(pop, neurite_type=NeuriteType.axon),
                       [])

    assert_array_equal(_nf.trunk_origin_elevations(pop, neurite_type=NeuriteType.apical_dendrite),
                       [])


@nt.raises(Exception)
def test_trunk_elevation_zero_norm_vector_raises():
    _nf.trunk_origin_elevations(SWC_NRN)


def test_sholl_crossings_simple():
    center = SIMPLE.soma.center
    radii = []
    nt.eq_(list(_nf.sholl_crossings(SIMPLE, center, radii=radii)),
           [])

    radii = [1.0]
    nt.eq_([2],
           list(_nf.sholl_crossings(SIMPLE, center, radii=radii)))

    radii = [1.0, 5.1]
    nt.eq_([2, 4],
           list(_nf.sholl_crossings(SIMPLE, center, radii=radii)))

    radii = [1., 4., 5.]
    nt.eq_([2, 4, 5],
           list(_nf.sholl_crossings(SIMPLE, center, radii=radii)))


def load_swc(string):
    with tempfile.NamedTemporaryFile(prefix='test_neuron_func', mode='w', suffix='.swc') as fd:
        fd.write(string)
        fd.flush()
        return load_neuron(fd.name)


def test_sholl_analysis_custom():
    # recreate morphs from Fig 2 of
    # http://dx.doi.org/10.1016/j.jneumeth.2014.01.016
    radii = np.arange(10, 81, 10)
    center = 0, 0, 0
    morph_A = load_swc("""\
 1 1   0  0  0 1. -1
 2 3   0  0  0 1.  1
 3 3  80  0  0 1.  2
 4 4   0  0  0 1.  1
 5 4 -80  0  0 1.  4""")
    nt.eq_(list(_nf.sholl_crossings(morph_A, center, radii=radii)),
           [2, 2, 2, 2, 2, 2, 2, 2])

    morph_B = load_swc("""\
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
                       """)
    nt.eq_(list(_nf.sholl_crossings(morph_B, center, radii=radii)),
           [2, 2, 2, 10, 10, 0, 0, 0])

    morph_C = load_swc("""\
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
                       """)
    nt.eq_(list(_nf.sholl_crossings(morph_C, center, radii=radii)),
           [2, 2, 2, 2, 2, 2, 10, 10])
    # view.neuron(morph_C)[0].savefig('foo.png')
