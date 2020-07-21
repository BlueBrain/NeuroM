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
"""compare neurom.features features with values dumped from the original
neurom._point_neurite.features"""

import json
import warnings
from pathlib import Path
from itertools import chain

from nose import tools as nt

import neurom as nm
from neurom.core import Tree
from neurom.core.types import NeuriteType

# NOTE: The 'bf' alias is used in the fst/tests modules
# Do NOT change it.
# TODO: If other neurom.features are imported,
# the should use the aliasing used in fst/tests module files
from neurom.features import bifurcationfunc as _bf
from neurom.features import neuritefunc as _nrt
from neurom.features import neuronfunc as _nrn
from neurom.features import sectionfunc as _sec
from neurom.features.tests.utils import _close, _equal

DATA_PATH =  Path(__file__).parent.parent.parent.parent / 'test_data'
SWC_DATA_PATH = DATA_PATH / 'swc'
H5V1_DATA_PATH = DATA_PATH / 'h5/v1'
H5V2_DATA_PATH = DATA_PATH / 'h5/v2'
MORPH_FILENAME = 'Neuron.h5'
SWC_MORPH_FILENAME = 'Neuron.swc'

REF_NEURITE_TYPES = [NeuriteType.apical_dendrite, NeuriteType.basal_dendrite,
                     NeuriteType.basal_dendrite, NeuriteType.axon]

json_data = json.load(open(Path(DATA_PATH, 'dataset/point_neuron_feature_values.json')))


def get(feat, neurite_format, **kwargs):
    """using the values captured from the old point_neurite system."""
    neurite_type = str(kwargs.get('neurite_type', ''))
    return json_data[neurite_format][feat][neurite_type]


def i_chain2(trees, iterator_type=Tree.ipreorder, mapping=None, tree_filter=None):
    """Returns a mapped iterator to a collection of trees
    Provides access to all the elements of all the trees
    in one iteration sequence.
    Parameters:
        trees: iterator or iterable of tree objects
        iterator_type: type of the iteration (segment, section, triplet...)
        mapping: optional function to apply to the iterator's target.
        tree_filter: optional top level filter on properties of tree objects.
    """
    nrt = (trees if tree_filter is None
           else filter(tree_filter, trees))

    chain_it = chain.from_iterable(map(iterator_type, nrt))
    return chain_it if mapping is None else map(mapping, chain_it)


class SectionTreeBase(object):
    """Base class for section tree tests."""

    def setUp(self):
        self.ref_nrn = 'h5'
        self.ref_types = REF_NEURITE_TYPES

    def test_neurite_type(self):
        neurite_types = [n0.type for n0 in self.sec_nrn.neurites]
        nt.assert_equal(neurite_types, self.ref_types)

    def test_get_n_sections(self):
        nt.assert_equal(_nrt.n_sections(self.sec_nrn),
                        get('number_of_sections', self.ref_nrn)[0])

        for t in NeuriteType:
            actual = _nrt.n_sections(self.sec_nrn, neurite_type=t)
            nt.assert_equal(actual,
                            get('number_of_sections', self.ref_nrn, neurite_type=t)[0])

    def test_get_number_of_sections_per_neurite(self):
        _equal(_nrt.number_of_sections_per_neurite(self.sec_nrn),
               get('number_of_sections_per_neurite', self.ref_nrn))

        for t in NeuriteType:
            _equal(_nrt.number_of_sections_per_neurite(self.sec_nrn, neurite_type=t),
                   get('number_of_sections_per_neurite', self.ref_nrn, neurite_type=t))

    def test_get_n_segments(self):
        nt.assert_equal(_nrt.n_segments(self.sec_nrn), get('number_of_segments', self.ref_nrn)[0])
        for t in NeuriteType:
            nt.assert_equal(_nrt.n_segments(self.sec_nrn, neurite_type=t),
                            get('number_of_segments', self.ref_nrn, neurite_type=t)[0])

    def test_get_number_of_neurites(self):
        nt.assert_equal(_nrt.n_neurites(self.sec_nrn), get('number_of_neurites', self.ref_nrn)[0])
        for t in NeuriteType:
            nt.assert_equal(_nrt.n_neurites(self.sec_nrn, neurite_type=t),
                            get('number_of_neurites', self.ref_nrn, neurite_type=t)[0])

    def test_get_section_path_distances(self):
        _close(_nrt.section_path_lengths(self.sec_nrn), get('section_path_distances', self.ref_nrn))
        for t in NeuriteType:
            _close(_nrt.section_path_lengths(self.sec_nrn, neurite_type=t),
                   get('section_path_distances', self.ref_nrn, neurite_type=t))

        pl = [_sec.section_path_length(s) for s in i_chain2(self.sec_nrn_trees)]
        _close(pl, get('section_path_distances', self.ref_nrn))

    def test_get_soma_radius(self):
        nt.assert_equal(self.sec_nrn.soma.radius, get('soma_radii', self.ref_nrn)[0])

    def test_get_soma_surface_area(self):
        nt.assert_equal(_nrn.soma_surface_area(self.sec_nrn), get('soma_surface_areas', self.ref_nrn)[0])

    def test_get_soma_volume(self):
        nt.assert_equal(_nrn.soma_volume(self.sec_nrn), get('soma_volumes', self.ref_nrn)[0])

    def test_get_local_bifurcation_angles(self):
        _close(_nrt.local_bifurcation_angles(self.sec_nrn),
               get('local_bifurcation_angles', self.ref_nrn))

        for t in NeuriteType:
            _close(_nrt.local_bifurcation_angles(self.sec_nrn, neurite_type=t),
                   get('local_bifurcation_angles', self.ref_nrn, neurite_type=t))

        ba = [_bf.local_bifurcation_angle(b)
              for b in i_chain2(self.sec_nrn_trees, iterator_type=Tree.ibifurcation_point)]

        _close(ba, get('local_bifurcation_angles', self.ref_nrn))

    def test_get_remote_bifurcation_angles(self):
        _close(_nrt.remote_bifurcation_angles(self.sec_nrn),
               get('remote_bifurcation_angles', self.ref_nrn))

        for t in NeuriteType:
            _close(_nrt.remote_bifurcation_angles(self.sec_nrn, neurite_type=t),
                   get('remote_bifurcation_angles', self.ref_nrn, neurite_type=t))

        ba = [_bf.remote_bifurcation_angle(b)
              for b in i_chain2(self.sec_nrn_trees, iterator_type=Tree.ibifurcation_point)]

        _close(ba, get('remote_bifurcation_angles', self.ref_nrn))

    def test_get_section_radial_distances(self):
        _close(_nrt.section_radial_distances(self.sec_nrn),
               get('section_radial_distances', self.ref_nrn))

        for t in NeuriteType:
            _close(_nrt.section_radial_distances(self.sec_nrn, neurite_type=t),
                   get('section_radial_distances', self.ref_nrn, neurite_type=t))

    def test_get_trunk_origin_radii(self):
        _equal(_nrn.trunk_origin_radii(self.sec_nrn), get('trunk_origin_radii', self.ref_nrn))
        for t in NeuriteType:
            _equal(_nrn.trunk_origin_radii(self.sec_nrn, neurite_type=t),
                   get('trunk_origin_radii', self.ref_nrn, neurite_type=t))

    def test_get_trunk_section_lengths(self):
        _close(_nrn.trunk_section_lengths(self.sec_nrn), get('trunk_section_lengths', self.ref_nrn))
        for t in NeuriteType:
            _close(_nrn.trunk_section_lengths(self.sec_nrn, neurite_type=t),
                   get('trunk_section_lengths', self.ref_nrn, neurite_type=t))


class TestH5V1(SectionTreeBase):

    def setUp(self):
        super(TestH5V1, self).setUp()
        self.sec_nrn = nm.load_neuron(Path(H5V1_DATA_PATH, MORPH_FILENAME))
        self.sec_nrn_trees = [n.root_node for n in self.sec_nrn.neurites]

    # Overriding soma values as the same soma points in SWC and ASC have different
    # meanings. Hence leading to different values
    def test_get_soma_radius(self):
        nt.assert_equal(self.sec_nrn.soma.radius, 0.09249506049313666)

    def test_get_soma_surface_area(self):
        nt.assert_equal(_nrn.soma_surface_area(self.sec_nrn),
                        0.1075095256160432)

    def test_get_soma_volume(self):
        with warnings.catch_warnings(record=True):
            nt.assert_equal(_nrn.soma_volume(self.sec_nrn), 0.0033147000251481135)


class TestH5V2(SectionTreeBase):

    def setUp(self):
        super(TestH5V2, self).setUp()
        self.sec_nrn = nm.load_neuron(Path(H5V2_DATA_PATH, MORPH_FILENAME))
        self.sec_nrn_trees = [n.root_node for n in self.sec_nrn.neurites]

    # Overriding soma values as the same soma points in SWC and ASC have different
    # meanings. Hence leading to different values
    def test_get_soma_radius(self):
        nt.assert_equal(self.sec_nrn.soma.radius, 0.09249506049313666)

    def test_get_soma_surface_area(self):
        nt.assert_equal(_nrn.soma_surface_area(self.sec_nrn),
                        0.1075095256160432)

    def test_get_soma_volume(self):
        with warnings.catch_warnings(record=True):
            nt.assert_equal(_nrn.soma_volume(self.sec_nrn), 0.0033147000251481135)


class TestSWC(SectionTreeBase):

    def setUp(self):
        self.ref_nrn = 'swc'
        self.sec_nrn = nm.load_neuron(Path(SWC_DATA_PATH, SWC_MORPH_FILENAME))
        self.sec_nrn_trees = [n.root_node for n in self.sec_nrn.neurites]
        self.ref_types = [NeuriteType.axon,
                          NeuriteType.basal_dendrite,
                          NeuriteType.basal_dendrite,
                          NeuriteType.apical_dendrite,
                          ]
