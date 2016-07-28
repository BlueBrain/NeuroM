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

import sys

import numpy as np
from nose import tools as nt
from neurom.core.tree import is_root
from neurom.core.tree import is_leaf
from neurom.core.tree import is_forking_point
from neurom.core.tree import is_bifurcation_point
from neurom.core.tree import ipreorder
from neurom.core.tree import ipostorder
from neurom.core.tree import iupstream
from neurom.core.tree import iforking_point
from neurom.core.tree import ibifurcation_point
from neurom.core.tree import ileaf
from neurom.point_neurite.point_tree import PointTree
from neurom.point_neurite.point_tree import isegment
from neurom.point_neurite.point_tree import itriplet
from neurom.point_neurite.point_tree import isection
from neurom.point_neurite.point_tree import val_iter
from neurom.point_neurite.point_tree import i_branch_end_points
from copy import deepcopy

REF_TREE = PointTree(0)
REF_TREE.add_child(PointTree(11))
REF_TREE.add_child(PointTree(12))
REF_TREE.children[0].add_child(PointTree(111))
REF_TREE.children[0].add_child(PointTree(112))
REF_TREE.children[1].add_child(PointTree(121))
REF_TREE.children[1].add_child(PointTree(122))
REF_TREE.children[1].children[0].add_child(PointTree(1211))
REF_TREE.children[1].children[0].children[0].add_child(PointTree(12111))
REF_TREE.children[1].children[0].children[0].add_child(PointTree(12112))

REF_TREE2 = deepcopy(REF_TREE)
T1111 = REF_TREE2.children[0].children[0].add_child(PointTree(1111))
T11111 = T1111.add_child(PointTree(11111))
T11112 = T1111.add_child(PointTree(11112))
T11113 = T1111.add_child(PointTree(11113))

REF_TREE3 = PointTree(np.array([0.,0.,0.,1.,0.,0.,0.]))
REF_TREE3.add_child(PointTree(np.array([1.,1.,1.,1.,0.,0.,0.])))
REF_TREE3.add_child(PointTree(np.array([1.,1.,2.,1.,0.,0.,0.])))
REF_TREE3.children[0].add_child(PointTree(np.array([2.,2.,2.,1.,0.,0.,0.])))
REF_TREE3.children[0].add_child(PointTree(np.array([2.,2.,3.,1.,0.,0.,0.])))
REF_TREE3.children[1].add_child(PointTree(np.array([3.,3.,3.,1.,0.,0.,0.])))
REF_TREE3.children[1].add_child(PointTree(np.array([3.,3.,4.,1.,0.,0.,0.])))
REF_TREE3.children[1].children[0].add_child(PointTree(np.array([4.,4.,4.,1.,0.,0.,0.])))
REF_TREE3.children[1].children[0].children[0].add_child(PointTree(np.array([5.,5.,5.,1.,0.,0.,0.])))
REF_TREE3.children[1].children[0].children[0].add_child(PointTree(np.array([5.,5.,6.,1.,0.,0.,0.])))

def test_str():
    t = PointTree('hello')
    nt.ok_(str(t))


def test_instantiate_tree():
    t = PointTree('hello')
    nt.ok_(t.parent is None)
    nt.ok_(t.value == 'hello')
    nt.ok_(len(t.children) == 0)


def test_children():
    nt.ok_(REF_TREE.children[0].value == 11)
    nt.ok_(REF_TREE.children[1].value == 12)
    nt.ok_(REF_TREE.children[0].children[0].value == 111)
    nt.ok_(REF_TREE.children[0].children[1].value == 112)
    nt.ok_(REF_TREE.children[1].children[0].value == 121)
    nt.ok_(REF_TREE.children[1].children[1].value == 122)


def test_add_child():
    t = PointTree(0)
    t.add_child(PointTree(11))
    t.add_child(PointTree(22))
    nt.ok_(t.value == 0)
    nt.ok_(len(t.children) == 2)
    nt.ok_([i.value for i in t.children] == [11, 22])


def test_parent():
    t = PointTree(0)
    for i in xrange(10):
        t.add_child(PointTree(i))

    nt.ok_(len(t.children) == 10)

    for c in t.children:
        nt.ok_(c.parent is t)


def test_is_root_true():
    t = PointTree(0)
    nt.ok_(is_root(t))


def test_is_root_false():
    t = PointTree(0)
    t.add_child(PointTree(1))
    nt.ok_(not is_root(t.children[0]))


def test_is_leaf():
    nt.ok_(is_leaf(PointTree(0)))


def test_is_leaf_false():
    t = PointTree(0)
    t.add_child(PointTree(1))
    nt.ok_(not is_leaf(t))


def test_is_forking_point():
    t = PointTree(0)
    t.add_child(PointTree(1))
    t.add_child(PointTree(2))
    nt.ok_(is_forking_point(t))
    t.add_child(PointTree(3))
    nt.ok_(is_forking_point(t))


def test_is_forking_point_false():
    t = PointTree(0)
    nt.ok_(not is_forking_point(t))
    t.add_child(PointTree(1))
    nt.ok_(not is_forking_point(t))


def test_is_bifurcation_point():
    t = PointTree(0)
    t.add_child(PointTree(1))
    t.add_child(PointTree(2))
    nt.ok_(is_bifurcation_point(t))


def test_is_bifurcation_point_false():
    t = PointTree(0)
    nt.ok_(not is_bifurcation_point(t))
    t.add_child(PointTree(1))
    nt.ok_(not is_bifurcation_point(t))
    t.add_child(PointTree(2))
    t.add_child(PointTree(3))
    nt.ok_(not is_bifurcation_point(t))


def test_deep_iteration():
    root = t = PointTree(0)
    for i in range(1, sys.getrecursionlimit() + 2):
        child = PointTree(i)
        t.add_child(child)
        t = child
    list(ipreorder(root))
    list(ipostorder(root))
    list(iupstream(t))


def test_preorder_iteration():
    nt.ok_(list(val_iter(ipreorder(REF_TREE))) ==
           [0, 11, 111, 112, 12, 121, 1211, 12111, 12112, 122])
    nt.ok_(list(val_iter(ipreorder(REF_TREE.children[0]))) == [11, 111, 112])
    nt.ok_(list(val_iter(ipreorder(REF_TREE.children[1]))) ==
           [12, 121, 1211, 12111, 12112, 122])


def test_postorder_iteration():
    nt.ok_(list(val_iter(ipostorder(REF_TREE))) ==
           [111, 112, 11, 12111, 12112, 1211, 121, 122, 12, 0])
    nt.ok_(list(val_iter(ipostorder(REF_TREE.children[0]))) == [111, 112, 11])
    nt.ok_(list(val_iter(ipostorder(REF_TREE.children[1]))) ==
           [12111, 12112, 1211, 121, 122, 12])


def test_upstream_iteration():

    nt.ok_(list(val_iter(iupstream(REF_TREE))) == [0])
    nt.ok_(list(val_iter(iupstream(REF_TREE.children[0]))) == [11, 0])
    nt.ok_(list(val_iter(iupstream(REF_TREE.children[0].children[0]))) ==
           [111, 11, 0])
    nt.ok_(list(val_iter(iupstream(REF_TREE.children[0].children[1]))) ==
           [112, 11, 0])


    nt.ok_(list(val_iter(iupstream(REF_TREE.children[1]))) == [12, 0])
    nt.ok_(list(val_iter(iupstream(REF_TREE.children[1].children[0]))) ==
           [121, 12, 0])
    nt.ok_(list(val_iter(iupstream(REF_TREE.children[1].children[1]))) ==
           [122, 12, 0])


def test_segment_iteration():

    nt.assert_equal(list(val_iter(isegment(REF_TREE))),
           [(0, 11),(11, 111),(11, 112),
            (0, 12),(12, 121),(121,1211),
            (1211,12111),(1211,12112),(12, 122)])

    nt.assert_equal(list(val_iter(isegment(REF_TREE.children[0]))),
           [(0, 11), (11, 111),(11, 112)])

    nt.assert_equal(list(val_iter(isegment(REF_TREE.children[0].children[0]))),
                    [(11, 111)])

    nt.assert_equal(list(val_iter(isegment(REF_TREE.children[0].children[1]))),
                    [(11, 112)])

    nt.assert_equal(list(val_iter(isegment(REF_TREE.children[1]))),
           [(0, 12), (12, 121), (121, 1211),
            (1211, 12111), (1211, 12112), (12, 122)])

    nt.assert_equal(list(val_iter(isegment(REF_TREE.children[1].children[0]))),
                    [(12, 121), (121, 1211), (1211, 12111), (1211, 12112)])

    nt.assert_equal(list(val_iter(isegment(REF_TREE.children[1].children[1]))),
                    [(12, 122)])


def test_segment_upstream_iteration():
    leaves = [l for l in ileaf(REF_TREE2)]
    ref_paths = [
        [(1111, 11111), (111, 1111), (11, 111), (0, 11)],
        [(1111, 11112), (111, 1111), (11, 111), (0, 11)],
        [(1111, 11113), (111, 1111), (11, 111), (0, 11)],
        [(11, 112), (0, 11)],
        [(1211, 12111), (121, 1211), (12, 121), (0, 12)],
        [(1211, 12112), (121, 1211), (12, 121), (0, 12)],
        [(12, 122), (0, 12)]
    ]

    for l, ref in zip(leaves, ref_paths):
        nt.assert_equal([s for s in val_iter(isegment(l, iupstream))], ref)


def test_itriplet():

    ref = [[0, 11, 111], [0, 11, 112], [11, 111, 1111], [111, 1111, 11111],
           [111, 1111, 11112], [111, 1111, 11113],
           [0, 12, 121], [0, 12, 122], [12, 121, 1211],
           [121, 1211, 12111], [121, 1211, 12112]]

    nt.assert_equal(list([n.value for n in t]
                         for t in itriplet(REF_TREE2)),
                    ref)


def test_leaf_iteration():
    nt.ok_(list(val_iter(ileaf(REF_TREE))) == [111, 112, 12111, 12112, 122])
    nt.ok_(list(val_iter(ileaf(REF_TREE.children[0]))) == [111, 112])
    nt.ok_(list(val_iter(ileaf(REF_TREE.children[1]))) == [12111, 12112, 122])
    nt.ok_(list(val_iter(ileaf(REF_TREE.children[0].children[0]))) == [111])
    nt.ok_(list(val_iter(ileaf(REF_TREE.children[0].children[1]))) == [112])
    nt.ok_(list(val_iter(ileaf(REF_TREE.children[1].children[0]))) == [12111, 12112])
    nt.ok_(list(val_iter(ileaf(REF_TREE.children[1].children[1]))) == [122])


def test_iforking_point():
    nt.assert_equal([n.value for n in iforking_point(REF_TREE2)],
                    [0, 11, 1111, 12, 1211])


def test_iforking_point_preorder():
    nt.assert_equal([n.value for n in iforking_point(REF_TREE2, ipostorder)],
                    [1111, 11, 1211, 12, 0])


def test_iforking_point_upstream():
    leaves = [l for l in ileaf(REF_TREE2)]
    ref_paths = [
        [1111, 11, 0], [1111, 11, 0], [1111, 11, 0], [11, 0],
        [1211, 12, 0], [1211, 12, 0], [12, 0]
    ]

    for l, ref in zip(leaves, ref_paths):
        nt.assert_equal([s for s in val_iter(iforking_point(l, iupstream))], ref)


def test_valiter_forking_point():
    nt.ok_(list(val_iter(iforking_point(REF_TREE2))) ==
           [0, 11, 1111, 12, 1211])


def test_ibifurcation_point():
    nt.assert_equal([n.value for n in ibifurcation_point(REF_TREE2)],
                    [0, 11, 12, 1211])


def test_ibifurcation_point_postorder():
    nt.assert_equal([n.value for n in ibifurcation_point(REF_TREE2, ipostorder)],
                    [11, 1211, 12, 0])


def test_ibifurcation_point_upstream():
    leaves = [l for l in ileaf(REF_TREE2)]
    ref_paths = [
        [11, 0], [11, 0], [11, 0], [11, 0],
        [1211, 12, 0], [1211, 12, 0], [12, 0]
    ]

    for l, ref in zip(leaves, ref_paths):
        nt.assert_equal([s for s in val_iter(ibifurcation_point(l, iupstream))], ref)


def test_valiter_bifurcation_point():
    nt.ok_(list(val_iter(ibifurcation_point(REF_TREE2))) ==
           [0, 11, 12, 1211])


def test_section_iteration():
    REF_SECTIONS = ((0, 11), (11, 111, 1111), (1111, 11111), (1111, 11112),
                    (1111, 11113), (11, 112), (0, 12), (12, 121, 1211),
                    (1211, 12111), (1211, 12112), (12, 122))

    for i, s in enumerate(isection(REF_TREE2)):
        nt.assert_equal(REF_SECTIONS[i], tuple(tt.value for tt in s))


def test_section_upstream_iteration():
    leaves = [l for l in ileaf(REF_TREE2)]
    ref_paths = [
        [(1111, 11111), (11, 111, 1111), (0, 11)],
        [(1111, 11112), (11, 111, 1111), (0, 11)],
        [(1111, 11113), (11, 111, 1111), (0, 11)],
        [(11, 112), (0, 11)],
        [(1211, 12111), (12, 121, 1211), (0, 12)],
        [(1211, 12112), (12, 121, 1211), (0, 12)],
        [(12, 122), (0, 12)]]

    for l, ref in zip(leaves, ref_paths):
        nt.assert_equal([s for s in val_iter(isection(l, iupstream))], ref)



def test_branch_end_points():

    def _build_tuple(tree):
        return tuple(p for p in val_iter(i_branch_end_points(tree)))

    nt.assert_equal(_build_tuple(REF_TREE2), (11, 12))
    nt.assert_equal(_build_tuple(REF_TREE2), (11, 12))
    nt.assert_equal(_build_tuple(REF_TREE2.children[0]), (1111, 112))
    nt.assert_equal(_build_tuple(REF_TREE2.children[1]), (1211, 122))

    nt.assert_equal(_build_tuple(REF_TREE2.children[0].children[0]), (1111,))
    nt.assert_equal(len(_build_tuple(REF_TREE2.children[0].children[1])), 0)

    nt.assert_equal(_build_tuple(REF_TREE2.children[1].children[0]), (1211,))
    nt.assert_equal(len(_build_tuple(REF_TREE2.children[1].children[1])), 0)

    nt.assert_equal(_build_tuple(REF_TREE2.children[0].children[0].children[0]),
                    (11111, 11112, 11113))
