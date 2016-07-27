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

'''Test neurom._point_neurite.features.get and neurom.fst features compatibility'''

import os
import numpy as np
from nose import tools as nt
from neurom.core.types import NeuriteType
from neurom.core.population import Population
from neurom import fst
from neurom.point_neurite.io.utils import load_neuron
from neurom.point_neurite.features import get
from neurom.point_neurite import treefunc as mt


_PWD = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_PWD, '../../../test_data/h5/v1/Neuron.h5')
NRN_PATHS = [DATA_PATH] * 10


def _close(a, b, debug=False, rtol=1e-05, atol=1e-08):
    if debug:
        print '\na.shape: %s\nb.shape: %s\n' % (a.shape, b.shape)
        print '\na: %s\nb:%s\n' % (a, b)
        print '\na - b:%s\n' % (a - b)
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.allclose(a, b, rtol=rtol, atol=atol))


def _equal(a, b, debug=False):
    if debug:
        print '\na.shape: %s\nb.shape: %s\n' % (a.shape, b.shape)
        print '\na: %s\nb:%s\n' % (a, b)
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.alltrue(a == b))


class TestSectionTree(object):
    '''Base class for section tree tests'''

    def setUp(self):
        self.ref_pop = Population([load_neuron(f, mt.set_tree_type) for f in NRN_PATHS])
        self.fst_pop = Population([fst.load_neuron(f) for f in NRN_PATHS])
        self.ref_types = [t.type for t in self.ref_pop.neurites]

    def _check_neuron_feature(self, ftr, debug=False, rtol=1e-05, atol=1e-08):
        _close(fst.get(ftr, self.fst_pop), get(ftr, self.ref_pop),
               debug, rtol, atol)

    def _check_neurite_feature(self, ftr, debug=False, rtol=1e-05, atol=1e-08):
        self._check_neuron_feature(ftr, debug, rtol, atol)

        for t in NeuriteType:
            _close(fst.get(ftr, self.fst_pop, neurite_type=t),
                   get(ftr, self.ref_pop, neurite_type=t), debug, rtol, atol)

    def test_get_soma_radius(self):
        self._check_neuron_feature('soma_radii')

    def test_get_soma_surface_area(self):
        self._check_neuron_feature('soma_surface_areas')

    def test_neurite_type(self):
        neurite_types = [n0.type for n0 in self.fst_pop.neurites]
        nt.assert_equal(neurite_types, self.ref_types)
        nt.assert_equal(neurite_types, [n1.type for n1 in self.ref_pop.neurites])

    @nt.nottest
    def test_get_n_sections(self):
        self._check_neurite_feature('number_of_sections')

    def test_get_n_sections_per_neurite(self):
        self._check_neurite_feature('number_of_sections_per_neurite')

    @nt.nottest
    def test_get_n_segments(self):
        self._check_neurite_feature('number_of_segments')

    @nt.nottest
    def test_get_number_of_neurites(self):
        self._check_neurite_feature('number_of_neurites')

    @nt.nottest
    def test_get_number_of_bifurcations(self):
        self._check_neurite_feature('number_of_bifurcations')

    def test_get_section_lengths(self):
        self._check_neurite_feature('section_lengths')

    def test_get_section_path_distances(self):
        self._check_neurite_feature('section_path_distances')

    def test_get_segment_lengths(self):
        self._check_neurite_feature('segment_lengths', debug=False)

    def test_get_segment_radii(self):
        self._check_neurite_feature('segment_radii', debug=False)

    def test_get_segment_radial_distances(self):
        self._check_neurite_feature('segment_radial_distances', debug=False)

    def test_get_segment_taper_rates(self):
        self._check_neurite_feature('segment_taper_rates', debug=False)

    def test_get_segment_midpoint(self):

        for ntyp in fst.NEURITE_TYPES:
            pts = fst.get('segment_midpoints', self.fst_pop, neurite_type=ntyp)
            ref_xyz = (get('segment_x_coordinates', self.ref_pop, neurite_type=ntyp),
                       get('segment_y_coordinates', self.ref_pop, neurite_type=ntyp),
                       get('segment_z_coordinates', self.ref_pop, neurite_type=ntyp))

            for i in xrange(3):
                _equal(pts[:, i], ref_xyz[i])

    def test_get_local_bifurcation_angles(self):
        self._check_neurite_feature('local_bifurcation_angles')

    def test_get_remote_bifurcation_angles(self):
        self._check_neurite_feature('remote_bifurcation_angles')

    def test_get_section_radial_distances(self):
        self._check_neurite_feature('section_radial_distances')

    def test_get_trunk_origin_radii(self):
        self._check_neurite_feature('trunk_origin_radii')

    def test_get_trunk_section_lengths(self):
        self._check_neurite_feature('trunk_section_lengths')

    def test_trunk_origin_azimuths(self):
        self._check_neurite_feature('trunk_origin_azimuths')

    @nt.nottest
    def test_trunk_origin_elevations(self):
        self._check_neurite_feature('trunk_origin_elevations')

    def test_get_section_branch_orders(self):
        self._check_neurite_feature('section_branch_orders')

    def test_get_partition(self):
        self._check_neurite_feature('partition')

    def test_get_section_areas(self):
        self._check_neurite_feature('section_areas')

    def test_get_section_volumes(self):
        self._check_neurite_feature('section_volumes')

    @nt.nottest
    def test_get_total_length(self):
        self._check_neurite_feature('total_length')

    def test_get_total_length_per_neurite(self):
        self._check_neurite_feature('total_length_per_neurite')

    def test_get_principal_direction_extents(self):
        self._check_neurite_feature('principal_direction_extents', debug=False)
