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

'''Test neurom.ezy.Neuron'''

import os
import math
import numpy as np
from neurom import ezy
from neurom.core.types import TreeType
from neurom.exceptions import SomaError, NonConsecutiveIDsError
from nose import tools as nt
from neurom.core.dataformat import COLS
from neurom.core.tree import ipreorder, val_iter
from neurom.analysis.morphtree import i_section_length
from neurom.analysis.morphtree import i_segment_length
from neurom.analysis.morphtree import i_local_bifurcation_angle
from neurom.analysis.morphtree import i_remote_bifurcation_angle
from neurom.analysis.morphtree import i_section_radial_dist
from neurom.analysis.morphtree import i_section_path_length

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')

def test_construct_neuron():
    filename = os.path.join(SWC_PATH, 'Neuron.swc')
    ezy.load_neuron(filename)


@nt.raises(SomaError)
def test_construct_neuron_no_soma_raises_SomaError():
    filename = os.path.join(SWC_PATH, 'Single_apical_no_soma.swc')
    ezy.load_neuron(filename)


@nt.raises(NonConsecutiveIDsError)
def test_construct_neuron_non_consecutive_ids_raises_NonConsecutiveIDsError():
    filename = os.path.join(SWC_PATH, 'non_sequential_trunk_off_1_16pt.swc')
    ezy.load_neuron(filename)


class TestEzyNeuron(object):

    def setUp(self):
        self.filename = os.path.join(SWC_PATH, 'Neuron.swc')
        self.neuron = ezy.load_neuron(self.filename)

    def test_name(self):
        nt.assert_true(self.neuron.name == 'Neuron')

    def test_get_section_lengths(self):
        ref_seclen = []
        for t in self.neuron.neurites:
            ref_seclen.extend(ll for ll in i_section_length(t))

        seclen = self.neuron.get_section_lengths()
        nt.assert_equal(len(seclen), 84)
        nt.assert_true(np.all(seclen == ref_seclen))

        seclen = self.neuron.get_section_lengths(TreeType.all)
        nt.assert_equal(len(seclen), 84)
        nt.assert_true(np.all(seclen == ref_seclen))

    def test_get_section_lengths_axon(self):
        s = self.neuron.get_section_lengths(TreeType.axon)
        nt.assert_equal(len(s), 21)

    def test_get_section_lengths_basal(self):
        s = self.neuron.get_section_lengths(TreeType.basal_dendrite)
        nt.assert_equal(len(s), 42)

    def test_get_section_lengths_apical(self):
        s = self.neuron.get_section_lengths(TreeType.apical_dendrite)
        nt.assert_equal(len(s), 21)

    def test_get_section_lengths_invalid(self):
        s = self.neuron.get_section_lengths(TreeType.soma)
        nt.assert_equal(len(s), 0)
        s = self.neuron.get_section_lengths(TreeType.undefined)
        s = self.neuron.get_section_lengths(TreeType.soma)

    def test_get_segment_lengths(self):
        ref_seglen = []
        for t in self.neuron.neurites:
            ref_seglen.extend(ll for ll in i_segment_length(t))

        seglen = self.neuron.get_segment_lengths()
        nt.assert_equal(len(seglen), 840)
        nt.assert_true(np.all(seglen == ref_seglen))

        seglen = self.neuron.get_segment_lengths(TreeType.all)
        nt.assert_equal(len(seglen), 840)
        nt.assert_true(np.all(seglen == ref_seglen))

    def test_get_soma_radius(self):
        nt.assert_almost_equal(self.neuron.get_soma_radius(), 0.170710678)

    def test_get_soma_surface_area(self):
        area = 4 * math.pi * (self.neuron.get_soma_radius() ** 2)
        nt.assert_almost_equal(self.neuron.get_soma_surface_area(), area)

    def test_get_segment_lengths_axon(self):
        s = self.neuron.get_segment_lengths(TreeType.axon)
        nt.assert_equal(len(s), 210)

    def test_get_segment_lengths_basal(self):
        s = self.neuron.get_segment_lengths(TreeType.basal_dendrite)
        nt.assert_equal(len(s), 420)

    def test_get_segment_lengths_apical(self):
        s = self.neuron.get_segment_lengths(TreeType.apical_dendrite)
        nt.assert_equal(len(s), 210)

    def test_get_segment_lengths_invalid(self):
        s = self.neuron.get_segment_lengths(TreeType.soma)
        nt.assert_equal(len(s), 0)
        s = self.neuron.get_segment_lengths(TreeType.undefined)
        nt.assert_equal(len(s), 0)

    def test_get_local_bifurcation_angles(self):
        ref_local_bifangles = []
        for t in self.neuron.neurites:
            ref_local_bifangles.extend(a for a in i_local_bifurcation_angle(t))

        local_bifangles = self.neuron.get_local_bifurcation_angles()
        nt.assert_equal(len(local_bifangles), 40)
        nt.assert_true(np.all(local_bifangles == ref_local_bifangles))
        local_bifangles = self.neuron.get_local_bifurcation_angles(TreeType.all)
        nt.assert_equal(len(local_bifangles), 40)
        nt.assert_true(np.all(local_bifangles == ref_local_bifangles))

    def test_get_local_bifurcation_angles_axon(self):
        s = self.neuron.get_local_bifurcation_angles(TreeType.axon)
        nt.assert_equal(len(s), 10)

    def test_get_local_bifurcation_angles_basal(self):
        s = self.neuron.get_local_bifurcation_angles(TreeType.basal_dendrite)
        nt.assert_equal(len(s), 20)

    def test_get_local_bifurcation_angles_apical(self):
        s = self.neuron.get_local_bifurcation_angles(TreeType.apical_dendrite)
        nt.assert_equal(len(s), 10)

    def test_get_local_bifurcation_angles_invalid(self):
        s = self.neuron.get_local_bifurcation_angles(TreeType.soma)
        nt.assert_equal(len(s), 0)
        s = self.neuron.get_local_bifurcation_angles(TreeType.undefined)
        nt.assert_equal(len(s), 0)

    def test_get_remote_bifurcation_angles(self):
        ref_remote_bifangles = []
        for t in self.neuron.neurites:
            ref_remote_bifangles.extend(a for a in i_remote_bifurcation_angle(t))

        remote_bifangles = self.neuron.get_remote_bifurcation_angles()
        nt.assert_equal(len(remote_bifangles), 40)
        nt.assert_true(np.all(remote_bifangles == ref_remote_bifangles))
        remote_bifangles = self.neuron.get_remote_bifurcation_angles(TreeType.all)
        nt.assert_equal(len(remote_bifangles), 40)
        nt.assert_true(np.all(remote_bifangles == ref_remote_bifangles))

    def test_get_remote_bifurcation_angles_axon(self):
        s = self.neuron.get_remote_bifurcation_angles(TreeType.axon)
        nt.assert_equal(len(s), 10)

    def test_get_remote_bifurcation_angles_basal(self):
        s = self.neuron.get_remote_bifurcation_angles(TreeType.basal_dendrite)
        nt.assert_equal(len(s), 20)

    def test_get_remote_bifurcation_angles_apical(self):
        s = self.neuron.get_remote_bifurcation_angles(TreeType.apical_dendrite)
        nt.assert_equal(len(s), 10)

    def test_get_remote_bifurcation_angles_invalid(self):
        s = self.neuron.get_remote_bifurcation_angles(TreeType.soma)
        nt.assert_equal(len(s), 0)
        s = self.neuron.get_remote_bifurcation_angles(TreeType.undefined)
        nt.assert_equal(len(s), 0)


    def test_get_section_radial_distances_endpoint(self):
        ref_sec_rad_dist_start = []
        for t in self.neuron.neurites:
            ref_sec_rad_dist_start.extend(
                ll for ll in i_section_radial_dist(t, use_start_point=True))

        ref_sec_rad_dist = []
        for t in self.neuron.neurites:
            ref_sec_rad_dist.extend(ll for ll in i_section_radial_dist(t))

        rad_dists = self.neuron.get_section_radial_distances()
        nt.assert_true(ref_sec_rad_dist != ref_sec_rad_dist_start)
        nt.assert_equal(len(rad_dists), 84)
        nt.assert_true(np.all(rad_dists == ref_sec_rad_dist))

    def test_get_section_radial_distances_start_point(self):
        ref_sec_rad_dist_start = []
        for t in self.neuron.neurites:
            ref_sec_rad_dist_start.extend(
                ll for ll in i_section_radial_dist(t, use_start_point=True))

        rad_dists = self.neuron.get_section_radial_distances(use_start_point=True)
        nt.assert_equal(len(rad_dists), 84)
        nt.assert_true(np.all(rad_dists == ref_sec_rad_dist_start))

    def test_get_section_radial_axon(self):
        rad_dists = self.neuron.get_section_radial_distances(neurite_type=TreeType.axon)
        nt.assert_equal(len(rad_dists), 21)

    def test_get_section_path_distances_endpoint(self):
        ref_sec_path_len_start = []
        for t in self.neuron.neurites:
            ref_sec_path_len_start.extend(
                ll for ll in i_section_path_length(t, use_start_point=True))

        ref_sec_path_len = []
        for t in self.neuron.neurites:
            ref_sec_path_len.extend(ll for ll in i_section_path_length(t))

        path_lengths = self.neuron.get_section_path_distances()
        nt.assert_true(ref_sec_path_len != ref_sec_path_len_start)
        nt.assert_equal(len(path_lengths), 84)
        nt.assert_true(np.all(path_lengths == ref_sec_path_len))

    def test_get_section_path_distances_start_point(self):
        ref_sec_path_len_start = []
        for t in self.neuron.neurites:
            ref_sec_path_len_start.extend(
                ll for ll in i_section_path_length(t, use_start_point=True))

        path_lengths = self.neuron.get_section_path_distances(use_start_point=True)
        nt.assert_equal(len(path_lengths), 84)
        nt.assert_true(np.all(path_lengths == ref_sec_path_len_start))

    def test_get_section_path_distances_axon(self):
        path_lengths = self.neuron.get_section_path_distances(neurite_type=TreeType.axon)
        nt.assert_equal(len(path_lengths), 21)


    def test_get_n_sections(self):
        nt.assert_equal(self.neuron.get_n_sections(), 84)
        nt.assert_equal(self.neuron.get_n_sections(TreeType.all), 84)

    def test_get_n_sections_axon(self):
        nt.assert_equal(self.neuron.get_n_sections(TreeType.axon), 21)

    def test_get_n_sections_basal(self):
        nt.assert_equal(self.neuron.get_n_sections(TreeType.basal_dendrite), 42)

    def test_get_n_sections_apical(self):
        nt.assert_equal(self.neuron.get_n_sections(TreeType.apical_dendrite), 21)

    def test_get_n_sections_invalid(self):
        nt.assert_equal(self.neuron.get_n_sections(TreeType.soma), 0)
        nt.assert_equal(self.neuron.get_n_sections(TreeType.undefined), 0)

    def test_get_n_sections_per_neurite(self):
        nsecs = self.neuron.get_n_sections_per_neurite()
        nt.assert_equal(len(nsecs), 4)
        nt.assert_true(np.all(nsecs == [21, 21, 21, 21]))

    def test_get_n_sections_per_neurite_axon(self):
        nsecs = self.neuron.get_n_sections_per_neurite(TreeType.axon)
        nt.assert_equal(len(nsecs), 1)
        nt.assert_equal(nsecs, [21])

    def test_get_n_sections_per_neurite_basal(self):
        nsecs = self.neuron.get_n_sections_per_neurite(TreeType.basal_dendrite)
        nt.assert_equal(len(nsecs), 2)
        nt.assert_true(np.all(nsecs == [21, 21]))

    def test_get_n_sections_per_neurite_apical(self):
        nsecs = self.neuron.get_n_sections_per_neurite(TreeType.apical_dendrite)
        nt.assert_equal(len(nsecs), 1)
        nt.assert_true(np.all(nsecs == [21]))

    def test_get_n_neurites(self):
        nt.assert_equal(self.neuron.get_n_neurites(), 4)
        nt.assert_equal(self.neuron.get_n_neurites(TreeType.all), 4)
        nt.assert_equal(self.neuron.get_n_neurites(TreeType.axon), 1)
        nt.assert_equal(self.neuron.get_n_neurites(TreeType.basal_dendrite), 2)
        nt.assert_equal(self.neuron.get_n_neurites(TreeType.apical_dendrite), 1)
        nt.assert_equal(self.neuron.get_n_neurites(TreeType.soma), 0)
        nt.assert_equal(self.neuron.get_n_neurites(TreeType.undefined), 0)

    def test_iter_points(self):
        ref_point_radii = []
        for t in self.neuron.neurites:
            ref_point_radii.extend(p[COLS.R] for p in val_iter(ipreorder(t)))

        rads = [r for r in self.neuron.iter_points(lambda p: p[COLS.R])]
        nt.assert_true(np.all(ref_point_radii == rads))

    def test_view(self):
        # Neuron.plot simply forwards arguments to neurom.view.view
        # So simply check that calling is OK syntactically.
        ezy.view(self.neuron)
        ezy.view3d(self.neuron)

    def test_bounding_box(self):
        bbox = ((-40.328535157399998, -57.6001719972, -0.17071067811865476),
                (64.7472627179, 48.516262252300002, 54.204087967500001))
        nt.ok_(np.allclose(bbox, self.neuron.bounding_box()))
