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

'''Test neurom._point_neurite.features and neurom.fst features compatibility'''

import os
import numpy as np
from nose import tools as nt
import neurom as nm
from neurom.core.types import NeuriteType
from neurom import fst
from neurom.fst import _neuronfunc as _nrn
from neurom.fst import _neuritefunc as _nrt
from neurom.fst import sectionfunc as _sec
from neurom.fst import _bifurcationfunc as _bf
from neurom.core import Tree
from neurom.core.tree import i_chain2
from neurom.point_neurite.io.utils import load_neuron as load_pt_neuron
from neurom.point_neurite.features import get
from neurom.point_neurite import treefunc as mt



_PWD = os.path.dirname(os.path.abspath(__file__))
SWC_DATA_PATH = os.path.join(_PWD, '../../../test_data/swc')
H5V1_DATA_PATH = os.path.join(_PWD, '../../../test_data/h5/v1')
H5V2_DATA_PATH = os.path.join(_PWD, '../../../test_data/h5/v2')
MORPH_FILENAME = 'Neuron.h5'
SWC_MORPH_FILENAME = 'Neuron.swc'

# Arbitrarily use h5 v1 as reference always
REF_NRN = load_pt_neuron(os.path.join(H5V1_DATA_PATH, MORPH_FILENAME),
                         mt.set_tree_type)

REF_NEURITE_TYPES = [NeuriteType.apical_dendrite, NeuriteType.basal_dendrite,
                     NeuriteType.basal_dendrite, NeuriteType.axon]

def _close(a, b, debug=False):
    if debug:
        print('a: %s\nb:%s\n' % (a, b))
    nt.assert_equal(len(a), len(b))
    nt.ok_(np.allclose(a, b))


def _equal(a, b, debug=False):
    if debug:
        print('a: %s\nb:%s\n' % (a, b))
    nt.assert_equal(len(a), len(b))
    nt.ok_(np.alltrue(a == b))


class SectionTreeBase(object):
    '''Base class for section tree tests'''

    def setUp(self):
        self.ref_nrn = REF_NRN
        self.ref_types = REF_NEURITE_TYPES



    def test_neurite_type(self):

        neurite_types = [n0.type for n0 in self.sec_nrn.neurites]
        nt.assert_equal(neurite_types, self.ref_types)
        nt.assert_equal(neurite_types, [n1.type for n1 in self.ref_nrn.neurites])

    def test_get_n_sections(self):
        nt.assert_equal(_nrt.n_sections(self.sec_nrn.neurites[0]),
                        get('number_of_sections', self.ref_nrn.neurites[0])[0])

        nt.assert_equal(_nrt.n_sections(self.sec_nrn), get('number_of_sections', self.ref_nrn)[0])
        for t in NeuriteType:
            nt.assert_equal(_nrt.n_sections(self.sec_nrn, neurite_type=t),
                            get('number_of_sections', self.ref_nrn, neurite_type=t)[0])

    def test_get_n_sections_per_neurite(self):
        _equal(_nrt.n_sections_per_neurite(self.sec_nrn.neurites[0]),
               get('number_of_sections_per_neurite', self.ref_nrn.neurites[0]))

        _equal(_nrt.n_sections_per_neurite(self.sec_nrn),
               get('number_of_sections_per_neurite', self.ref_nrn))

        for t in NeuriteType:
            _equal(_nrt.n_sections_per_neurite(self.sec_nrn, neurite_type=t),
                   get('number_of_sections_per_neurite', self.ref_nrn, neurite_type=t))

    def test_get_n_segments(self):
        nt.assert_equal(_nrt.n_segments(self.sec_nrn.neurites[0]),
                        get('number_of_segments', self.ref_nrn.neurites[0])[0])

        nt.assert_equal(_nrt.n_segments(self.sec_nrn), get('number_of_segments', self.ref_nrn)[0])
        for t in NeuriteType:
            nt.assert_equal(_nrt.n_segments(self.sec_nrn, neurite_type=t),
                            get('number_of_segments', self.ref_nrn, neurite_type=t)[0])

    def test_get_number_of_neurites(self):
        nt.assert_equal(_nrt.n_neurites(self.sec_nrn.neurites[0]),
                        get('number_of_neurites', self.ref_nrn.neurites[0])[0])

        nt.assert_equal(_nrt.n_neurites(self.sec_nrn), get('number_of_neurites', self.ref_nrn)[0])
        for t in NeuriteType:
            nt.assert_equal(_nrt.n_neurites(self.sec_nrn, neurite_type=t),
                            get('number_of_neurites', self.ref_nrn, neurite_type=t)[0])

    def test_get_section_path_distances(self):
        _close(_nrt.section_path_lengths(self.sec_nrn.neurites[0]),
               get('section_path_distances', self.ref_nrn.neurites[0]))

        _close(_nrt.section_path_lengths(self.sec_nrn), get('section_path_distances', self.ref_nrn))
        for t in NeuriteType:
            _close(_nrt.section_path_lengths(self.sec_nrn, neurite_type=t),
                   get('section_path_distances', self.ref_nrn, neurite_type=t))

        pl = [_sec.section_path_length(s) for s in i_chain2(self.sec_nrn_trees)]
        _close(pl, get('section_path_distances', self.ref_nrn))

    @nt.nottest
    def test_get_segment_lengths(self):
        _equal(_nrt.segment_lengths(self.sec_nrn.neurites[0]),
               get('segment_lengths', self.ref_nrn.neurites[0]))

        _equal(_nrt.segment_lengths(self.sec_nrn), get('segment_lengths', self.ref_nrn))
        for t in NeuriteType:
            _equal(_nrt.segment_lengths(self.sec_nrn, neurite_type=t),
                   get('segment_lengths', self.ref_nrn, neurite_type=t))

    def test_get_soma_radius(self):
        nt.assert_equal(self.sec_nrn.soma.radius, get('soma_radii', self.ref_nrn)[0])

    def test_get_soma_surface_area(self):
        nt.assert_equal(fst._nrn.soma_surface_area(self.sec_nrn), get('soma_surface_areas', self.ref_nrn)[0])

    def test_get_local_bifurcation_angles(self):
        _close(_nrt.local_bifurcation_angles(self.sec_nrn.neurites[0]),
               get('local_bifurcation_angles', self.ref_nrn.neurites[0]))

        _close(_nrt.local_bifurcation_angles(self.sec_nrn),
               get('local_bifurcation_angles', self.ref_nrn))

        for t in NeuriteType:
            _close(_nrt.local_bifurcation_angles(self.sec_nrn, neurite_type=t),
                   get('local_bifurcation_angles', self.ref_nrn, neurite_type=t))

        ba = [_bf.local_bifurcation_angle(b)
              for b in i_chain2(self.sec_nrn_trees, iterator_type=Tree.ibifurcation_point)]

        _close(ba, get('local_bifurcation_angles', self.ref_nrn))

    def test_get_remote_bifurcation_angles(self):
        _close(_nrt.remote_bifurcation_angles(self.sec_nrn.neurites[0]),
               get('remote_bifurcation_angles', self.ref_nrn.neurites[0]))

        _close(_nrt.remote_bifurcation_angles(self.sec_nrn),
               get('remote_bifurcation_angles', self.ref_nrn))

        for t in NeuriteType:
            _close(_nrt.remote_bifurcation_angles(self.sec_nrn, neurite_type=t),
                   get('remote_bifurcation_angles', self.ref_nrn, neurite_type=t))

        ba = [_bf.remote_bifurcation_angle(b)
              for b in i_chain2(self.sec_nrn_trees, iterator_type=Tree.ibifurcation_point)]

        _close(ba, get('remote_bifurcation_angles', self.ref_nrn))

    def test_get_section_radial_distances(self):
        _close(_nrt.section_radial_distances(self.sec_nrn.neurites[0]),
               get('section_radial_distances', self.ref_nrn.neurites[0]))

        _close(_nrt.section_radial_distances(self.sec_nrn),
               get('section_radial_distances', self.ref_nrn))

        for t in NeuriteType:
            _close(_nrt.section_radial_distances(self.sec_nrn, neurite_type=t),
                   get('section_radial_distances', self.ref_nrn, neurite_type=t))

    def test_get_trunk_origin_radii(self):
        _equal(fst._nrn.trunk_origin_radii(self.sec_nrn), get('trunk_origin_radii', self.ref_nrn))
        for t in NeuriteType:
            _equal(_nrn.trunk_origin_radii(self.sec_nrn, neurite_type=t),
                   get('trunk_origin_radii', self.ref_nrn, neurite_type=t))

    def test_get_trunk_section_lengths(self):
        _equal(_nrn.trunk_section_lengths(self.sec_nrn), get('trunk_section_lengths', self.ref_nrn))
        for t in NeuriteType:
            _equal(_nrn.trunk_section_lengths(self.sec_nrn, neurite_type=t),
                   get('trunk_section_lengths', self.ref_nrn, neurite_type=t))


class TestH5V1(SectionTreeBase):

    def setUp(self):
        super(TestH5V1, self).setUp()
        self.sec_nrn = nm.load_neuron(os.path.join(H5V1_DATA_PATH, MORPH_FILENAME))
        self.sec_nrn_trees = [n.root_node for n in self.sec_nrn.neurites]


class TestH5V2(SectionTreeBase):

    def setUp(self):
        super(TestH5V2, self).setUp()
        self.sec_nrn = nm.load_neuron(os.path.join(H5V2_DATA_PATH, MORPH_FILENAME))
        self.sec_nrn_trees = [n.root_node for n in self.sec_nrn.neurites]


class TestSWC(SectionTreeBase):

    def setUp(self):
        self.ref_nrn = load_pt_neuron(os.path.join(SWC_DATA_PATH, SWC_MORPH_FILENAME),
                                   mt.set_tree_type)
        self.sec_nrn = nm.load_neuron(os.path.join(SWC_DATA_PATH, SWC_MORPH_FILENAME))
        self.sec_nrn_trees = [n.root_node for n in self.sec_nrn.neurites]
        self.ref_types = [n.type for n in self.ref_nrn.neurites]
