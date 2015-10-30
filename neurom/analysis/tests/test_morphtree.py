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

from nose import tools as nt
import os
from neurom.core.tree import Tree
import neurom.core.tree as tr
from neurom.core.types import TreeType
from neurom.io.utils import make_neuron
from neurom.core.neuron import make_soma
from neurom.io.readers import load_data
from neurom.analysis.morphmath import angle_3points
from neurom.analysis.morphtree import path_length
from neurom.analysis.morphtree import branch_order
from neurom.analysis.morphtree import i_segment_length
from neurom.analysis.morphtree import i_segment_volume
from neurom.analysis.morphtree import i_segment_area
from neurom.analysis.morphtree import i_segment_radius
from neurom.analysis.morphtree import i_segment_radial_dist
from neurom.analysis.morphtree import i_segment_meander_angle
from neurom.analysis.morphtree import find_tree_type
from neurom.analysis.morphtree import set_tree_type
from neurom.analysis.morphtree import get_tree_type
from neurom.analysis.morphtree import i_section_length
from neurom.analysis.morphtree import i_section_path_length
from neurom.analysis.morphtree import i_section_radial_dist
from neurom.analysis.morphtree import i_local_bifurcation_angle
from neurom.analysis.morphtree import i_remote_bifurcation_angle
from neurom.analysis.morphtree import n_sections
from neurom.analysis.morphtree import n_segments
from neurom.analysis.morphtree import n_bifurcations
from neurom.analysis.morphtree import n_terminations
from neurom.analysis.morphtree import trunk_radius
from neurom.analysis.morphtree import trunk_length
from neurom.analysis.morphtree import trunk_direction
from neurom.analysis.morphtree import trunk_elevation
from neurom.analysis.morphtree import trunk_azimuth
from neurom.analysis.morphtree import get_bounding_box
import math
import numpy as np
from itertools import izip

DATA_PATH = './test_data'
SWC_PATH = os.path.join(DATA_PATH, 'swc/')

data    = load_data(SWC_PATH + 'Neuron.swc')
neuron0 = make_neuron(data)
tree0   = neuron0.neurites[0]
tree_types = [TreeType.axon,
              TreeType.basal_dendrite,
              TreeType.basal_dendrite,
              TreeType.apical_dendrite]


# Mock tree holding integers, not points
MOCK_TREE = Tree(0)
MOCK_TREE.add_child(Tree(11))
MOCK_TREE.add_child(Tree(12))
MOCK_TREE.children[0].add_child(Tree(111))
MOCK_TREE.children[0].add_child(Tree(112))
MOCK_TREE.children[1].add_child(Tree(121))
MOCK_TREE.children[1].add_child(Tree(122))
MOCK_TREE.children[1].children[0].add_child(Tree(1211))
MOCK_TREE.children[1].children[0].children[0].add_child(Tree(12111))
MOCK_TREE.children[1].children[0].children[0].add_child(Tree(12112))
T1111 = MOCK_TREE.children[0].children[0].add_child(Tree(1111))
T11111 = T1111.add_child(Tree(11111))
T11112 = T1111.add_child(Tree(11112))
T11113 = T1111.add_child(Tree(11113))


def _form_neuron_tree():
    p = [0.0, 0.0, 0.0, 1.0, 1, 1, 2]
    T = Tree(p)
    T1 = T.add_child(Tree([0.0, 1.0, 0.0, 1.0, 1, 1, 2]))
    T2 = T1.add_child(Tree([0.0, 2.0, 0.0, 1.0, 1, 1, 2]))
    T3 = T2.add_child(Tree([0.0, 4.0, 0.0, 2.0, 1, 1, 2]))
    T4 = T3.add_child(Tree([0.0, 5.0, 0.0, 2.0, 1, 1, 2]))
    T5 = T4.add_child(Tree([2.0, 5.0, 0.0, 1.0, 1, 1, 2]))
    T6 = T4.add_child(Tree([0.0, 5.0, 2.0, 1.0, 1, 1, 2]))
    T7 = T5.add_child(Tree([3.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T8 = T7.add_child(Tree([4.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T9 = T6.add_child(Tree([0.0, 5.0, 3.0, 0.75, 1, 1, 2]))
    T10 = T9.add_child(Tree([0.0, 6.0, 3.0, 0.75, 1, 1, 2]))
    return T


def _form_simple_tree():
    p = [0.0, 0.0, 0.0, 1.0, 1, 1, 1]
    T = Tree(p)
    T1 = T.add_child(Tree([0.0, 2.0, 0.0, 1.0, 1, 1, 1]))
    T2 = T1.add_child(Tree([0.0, 4.0, 0.0, 1.0, 1, 1, 1]))
    T3 = T2.add_child(Tree([0.0, 6.0, 0.0, 1.0, 1, 1, 1]))
    T4 = T3.add_child(Tree([0.0, 8.0, 0.0, 1.0, 1, 1, 1]))

    T5 = T.add_child(Tree([0.0, 0.0, 2.0, 1.0, 1, 1, 1]))
    T6 = T5.add_child(Tree([0.0, 0.0, 4.0, 1.0, 1, 1, 1]))
    T7 = T6.add_child(Tree([0.0, 0.0, 6.0, 1.0, 1, 1, 1]))
    T8 = T7.add_child(Tree([0.0, 0.0, 8.0, 1.0, 1, 1, 1]))

    return T


NEURON_TREE = _form_neuron_tree()
SIMPLE_TREE = _form_simple_tree()


def _form_branching_tree():
    p = [0.0, 0.0, 0.0, 1.0, 1, 1, 2]
    T = Tree(p)
    T1 = T.add_child(Tree([0.0, 1.0, 0.0, 1.0, 1, 1, 2]))
    T2 = T1.add_child(Tree([0.0, 2.0, 0.0, 1.0, 1, 1, 2]))
    T3 = T2.add_child(Tree([0.0, 4.0, 0.0, 2.0, 1, 1, 2]))
    T4 = T3.add_child(Tree([0.0, 5.0, 0.0, 2.0, 1, 1, 2]))
    T5 = T4.add_child(Tree([2.0, 5.0, 0.0, 1.0, 1, 1, 2]))
    T6 = T4.add_child(Tree([0.0, 5.0, 2.0, 1.0, 1, 1, 2]))
    T7 = T5.add_child(Tree([3.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T8 = T7.add_child(Tree([4.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T9 = T6.add_child(Tree([0.0, 5.0, 3.0, 0.75, 1, 1, 2]))
    T10 = T9.add_child(Tree([0.0, 6.0, 3.0, 0.75, 1, 1, 2]))
    T11 = T9.add_child(Tree([0.0, 6.0, 4.0, 0.75, 1, 1, 2]))
    T33 = T3.add_child(Tree([1.0, 5.0, 0.0, 2.0, 1, 1, 2]))
    T331 = T33.add_child(Tree([15.0, 15.0, 0.0, 2.0, 1, 1, 2]))
    return T

BRANCHING_TREE = _form_branching_tree()


def test_segment_lengths():

    T = NEURON_TREE

    lg = [l for l in i_segment_length(T)]

    nt.assert_equal(lg, [1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0])


def test_segment_volumes():

    T = NEURON_TREE

    sv = (l/math.pi for l in i_segment_volume(T))

    ref = (1.0, 1.0, 4.6666667, 4.0, 4.6666667, 0.7708333,
           0.5625, 4.6666667, 0.7708333, 0.5625)

    for a, b in izip(sv, ref):
        nt.assert_almost_equal(a, b)


def test_segment_areas():

    T = NEURON_TREE

    sa = (l/math.pi for l in i_segment_area(T))

    ref = (2.0, 2.0, 6.7082039, 4.0, 6.7082039, 1.8038587,
           1.5, 6.7082039, 1.8038587, 1.5)

    for a, b in izip(sa, ref):
        nt.assert_almost_equal(a, b)



def test_segment_radiuss():

    T = NEURON_TREE

    rad = [r for r in i_segment_radius(T)]

    nt.assert_equal(rad,
                    [1.0, 1.0, 1.5, 2.0, 1.5, 0.875, 0.75, 1.5, 0.875, 0.75])


def test_segment_radial_dists():
    T = SIMPLE_TREE

    p= [0.0, 0.0, 0.0]

    rd = [d for d in i_segment_radial_dist(p,T)]

    nt.assert_equal(rd, [1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0])


def test_segment_path_length():
    leaves = [l for l in tr.ileaf(NEURON_TREE)]
    for l in leaves:
        nt.ok_(path_length(l) == 9)

    leaves = [l for l in tr.ileaf(SIMPLE_TREE)]
    for l in leaves:
        nt.ok_(path_length(l) == 8)


def test_find_tree_type():
    for en_tree, test_tree in enumerate(neuron0.neurites):
        nt.ok_(find_tree_type(test_tree) == tree_types[en_tree])


def test_set_tree_type():
    for en_tree, test_tree in enumerate(neuron0.neurites):
        set_tree_type(test_tree)
        nt.ok_(test_tree.type == tree_types[en_tree])


def test_get_tree_type():
    for en_tree, test_tree in enumerate(neuron0.neurites):
        if hasattr(test_tree, 'type'):
            del test_tree.type
        # tree.type should be computed here.
        nt.ok_(get_tree_type(test_tree) == tree_types[en_tree])
        find_tree_type(test_tree)
        # tree.type should already exists here, from previous action.
        nt.ok_(get_tree_type(test_tree) == tree_types[en_tree])

def test_i_section_length():
    T = SIMPLE_TREE
    nt.assert_equal([l for l in i_section_length(T)], [8.0, 8.0])
    T2 = NEURON_TREE
    nt.ok_([l for l in i_section_length(T2)] == [5.0, 4.0, 4.0])


def test_i_section_radial_dists():
    T1 = SIMPLE_TREE
    T2 = NEURON_TREE

    p0 = [0.0, 0.0, 0.0]

    nt.assert_equal([d for d in i_section_radial_dist(T1)],
                    [8.0, 8.0])

    nt.assert_equal([d for d in i_section_radial_dist(T1, p0)],
                    [8.0, 8.0])

    nt.assert_equal([d for d in i_section_radial_dist(T1, use_start_point=True)],
                    [0.0, 0.0])

    nt.assert_equal([d for d in i_section_radial_dist(T1, p0, use_start_point=True)],
                    [0.0, 0.0])

    nt.assert_true(np.allclose([d for d in i_section_radial_dist(T2)],
                               [5.0, 6.4031, 6.7082]))

    nt.assert_true(np.allclose([d for d in i_section_radial_dist(T2, p0)],
                               [5.0, 6.4031, 6.7082]))

    nt.assert_equal([d for d in i_section_radial_dist(T2, use_start_point=True)],
                    [0.0, 5.0, 5.0])

    nt.assert_equal([d for d in i_section_radial_dist(T2, p0, use_start_point=True)],
                    [0.0, 5.0, 5.0])


def test_i_section_path_length():
    T1 = SIMPLE_TREE
    T2 = NEURON_TREE

    nt.assert_equal([d for d in i_section_path_length(T1)],
                    [8.0, 8.0])

    nt.assert_equal([d for d in i_section_path_length(T1, use_start_point=True)],
                    [0.0, 0.0])

    nt.assert_equal([d for d in i_section_path_length(T2)], [5.0, 9.0, 9.0])

    nt.assert_equal([d for d in i_section_path_length(T2, use_start_point=True)],
                    [0.0, 5.0, 5.0])


def test_i_segment_meander_angles():
    T = NEURON_TREE
    ref = [math.pi * a for a in (1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5)]
    for i, m in enumerate(i_segment_meander_angle(T)):
        nt.assert_almost_equal(m, ref[i])


def test_i_local_bifurcation_angles():
    T = BRANCHING_TREE
    ref = (0.25, 0.5, 0.25)  # ref angles in pi radians
    for i, b in enumerate(i_local_bifurcation_angle(T)):
        nt.assert_almost_equal(b / math.pi, ref[i])


def test_i_remote_bifurcation_angles():
    T = BRANCHING_TREE

    # (fork point, end point, end point) tuples calculated by hand
    # from BRANCHING_TREE
    refs = (((0, 4, 0), (0, 5, 0), (15, 15, 0)),
            ((0, 5, 0), (4, 5, 0), (0, 5, 3)),
            ((0, 5, 3), (0, 6, 3), (0, 6, 4)))

    ref_angles = tuple(angle_3points(p[0], p[1], p[2]) / math.pi for p in refs)
    ref = (0.2985898, 0.5, 0.25)  # ref angles in pi radians
    # sanity check
    for i, b in enumerate(ref_angles):
        nt.assert_almost_equal(b, ref[i])

    for i, b in enumerate(i_remote_bifurcation_angle(T)):
        nt.assert_almost_equal(b / math.pi, ref_angles[i])
        nt.assert_almost_equal(b / math.pi, ref[i])


def test_n_sections():
    T = SIMPLE_TREE
    nt.ok_(n_sections(T) == 2)
    T2 = NEURON_TREE
    nt.ok_(n_sections(T2) == 3)


def test_n_segments():
    nt.ok_(n_segments(tree0) == 210)


def test_n_bifurcations():
    nt.ok_(n_bifurcations(tree0) == 10)


def test_n_terminations():
    nt.ok_(n_terminations(tree0) == 11)


def test_trunk_radius():
    t = Tree((0, 0, 0, 42))
    t.add_child(Tree((1, 0, 0, 4)))
    nt.assert_equal(trunk_radius(t), 42.0)


def test_trunk_length():
    t = Tree((0, 0, 0, 42))
    tt = t.add_child(Tree((10, 0, 0, 4)))
    tt.add_child(Tree((10, 15, 0, 4)))
    nt.assert_almost_equal(trunk_length(t), 25.0)


def test_trunk_radius_length_point_tree():
    t = Tree((0, 0, 0, 42))
    nt.assert_equal(trunk_length(t), 0.0)


def test_trunk_direction():
    t = Tree((1, 0, 0, 2))
    s = make_soma([[0, 0, 0, 4]])
    nt.ok_(np.allclose(trunk_direction(t, s), np.array([1, 0, 0])))


def test_trunk_elevation():
    t = Tree((1, 0, 0, 2))
    s = make_soma([[0, 0, 0, 4]])
    nt.assert_equal(trunk_elevation(t, s), 0.0)
    t = Tree((0, 1, 0, 2))
    nt.assert_equal(trunk_elevation(t, s),  np.pi/2)
    t = Tree((0, -1, 0, 2))
    nt.assert_equal(trunk_elevation(t, s),  -np.pi/2)
    t = Tree((0, 0, 0, 2))
    try:
        trunk_elevation(t, s)
        nt.ok_(False)
    except ValueError:
        nt.ok_(True)


def test_trunk_azimuth():
    t = Tree((0, 0, 0, 2))
    s = make_soma([[0, 0, 1, 4]])
    nt.assert_equal(trunk_azimuth(t, s), -np.pi/2)
    s = make_soma([[0, 0, -1, 4]])
    nt.assert_equal(trunk_azimuth(t, s), np.pi/2)
    s = make_soma([[0, 0, 0, 4]])
    nt.assert_equal(trunk_azimuth(t, s), 0.0)
    s = make_soma([[-1, 0, -1, 4]])
    nt.assert_equal(trunk_azimuth(t, s), np.pi/4)
    s = make_soma([[-1, 0, 0, 4]])
    nt.assert_equal(trunk_azimuth(t, s), 0.0)
    s = make_soma([[1, 0, 0, 4]])
    nt.assert_equal(trunk_azimuth(t, s), np.pi)



def test_get_bounding_box():
    box = np.array([[-33.25305769, -57.600172  ,   0.        ],
                    [  0.        ,   0.        ,  49.70137991]])
    bb = get_bounding_box(tree0)
    nt.ok_(np.allclose(bb, box))


def test_branch_order():

    branch_order_map = {
        (0, 11): 0,
        (11, 111, 1111): 1,
        (1111, 11111): 2,
        (1111, 11112): 2,
        (1111, 11113): 2,
        (11, 112): 1,
        (0, 12): 0,
        (12, 121, 1211): 1,
        (1211, 12111): 2,
        (1211, 12112): 2,
        (12, 122): 1
    }
    for sec in tr.isection(MOCK_TREE):
        nt.assert_equal(branch_order(sec),
                        branch_order_map[tuple(p for p in tr.val_iter(sec))])
