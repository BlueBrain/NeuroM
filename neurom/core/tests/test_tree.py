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
from nose import tools as nt
from neurom.core.tree import Tree

REF_TREE = Tree()
T11 = REF_TREE.add_child(Tree())
T12 = REF_TREE.add_child(Tree())
T111 = REF_TREE.children[0].add_child(Tree())
T112 = REF_TREE.children[0].add_child(Tree())
T121 = REF_TREE.children[1].add_child(Tree())
T122 = REF_TREE.children[1].add_child(Tree())
T1211 = REF_TREE.children[1].children[0].add_child(Tree())
T12111 = REF_TREE.children[1].children[0].children[0].add_child(Tree())
T12112 = REF_TREE.children[1].children[0].children[0].add_child(Tree())

REF_TREE2 = Tree()
T11_ = REF_TREE2.add_child(Tree())
T12_ = REF_TREE2.add_child(Tree())
T111_ = REF_TREE2.children[0].add_child(Tree())
T112_ = REF_TREE2.children[0].add_child(Tree())
T121_ = REF_TREE2.children[1].add_child(Tree())
T122_ = REF_TREE2.children[1].add_child(Tree())
T1211_ = REF_TREE2.children[1].children[0].add_child(Tree())
T12111_ = REF_TREE2.children[1].children[0].children[0].add_child(Tree())
T12112_ = REF_TREE2.children[1].children[0].children[0].add_child(Tree())
T1111_ = REF_TREE2.children[0].children[0].add_child(Tree())
T11111_ = T1111_.add_child(Tree())
T11112_ = T1111_.add_child(Tree())
T11113_ = T1111_.add_child(Tree())


def test_instantiate_tree():
    t = Tree()
    nt.ok_(t.parent is None)
    nt.ok_(len(t.children) == 0)


def test_add_child():
    t = Tree()
    ch11 = t.add_child(Tree())
    ch22 = t.add_child(Tree())
    nt.ok_(len(t.children) == 2)
    nt.ok_(t.children == [ch11, ch22])


def test_parent():
    t = Tree()
    for i in range(10):
        t.add_child(Tree())

    nt.ok_(len(t.children) == 10)

    for c in t.children:
        nt.ok_(c.parent is t)


def test_is_root_true():
    t = Tree()
    nt.ok_(Tree.is_root(t))
    nt.ok_(t.is_root())


def test_is_root_false():
    t = Tree()
    t.add_child(Tree())
    nt.ok_(not t.children[0].is_root())


def test_is_leaf():
    nt.ok_(Tree().is_leaf())


def test_is_leaf_false():
    t = Tree()
    t.add_child(Tree())
    nt.ok_(not t.is_leaf())


def test_is_forking_point():
    t = Tree()
    t.add_child(Tree())
    t.add_child(Tree())
    nt.ok_(t.is_forking_point())
    t.add_child(Tree())
    nt.ok_(t.is_forking_point())


def test_is_forking_point_false():
    t = Tree()
    nt.ok_(not t.is_forking_point())
    t.add_child(Tree())
    nt.ok_(not t.is_forking_point())


def test_is_bifurcation_point():
    t = Tree()
    t.add_child(Tree())
    t.add_child(Tree())
    nt.ok_(t.is_bifurcation_point())


def test_is_bifurcation_point_false():
    t = Tree()
    nt.ok_(not t.is_bifurcation_point())
    t.add_child(Tree())
    nt.ok_(not t.is_bifurcation_point())
    t.add_child(Tree())
    t.add_child(Tree())
    nt.ok_(not t.is_bifurcation_point())


def test_deep_iteration():
    root = t = Tree()
    for i in range(1, sys.getrecursionlimit() + 2):
        child = Tree()
        t.add_child(child)
        t = child

    list(root.ipreorder())
    list(root.ipostorder())
    list(t.iupstream())


def test_preorder_iteration():
    nt.ok_(list(REF_TREE.ipreorder()) ==
           [REF_TREE, T11, T111, T112, T12, T121, T1211, T12111, T12112, T122])

    nt.ok_(list(REF_TREE.children[0].ipreorder()) == [T11, T111, T112])

    nt.ok_(list(REF_TREE.children[1].ipreorder()) ==
           [T12, T121, T1211, T12111, T12112, T122])


def test_postorder_iteration():
    nt.ok_(list(REF_TREE.ipostorder()) ==
           [T111, T112, T11, T12111, T12112, T1211, T121, T122, T12, REF_TREE])
    nt.ok_(list(REF_TREE.children[0].ipostorder()) == [T111, T112, T11])
    nt.ok_(list(REF_TREE.children[1].ipostorder()) ==
           [T12111, T12112, T1211, T121, T122, T12])


def test_upstream_iteration():

    nt.ok_(list(REF_TREE.iupstream()) == [REF_TREE])
    nt.ok_(list(REF_TREE.children[0].iupstream()) == [T11, REF_TREE])
    nt.ok_(list(REF_TREE.children[0].children[0].iupstream()) ==
           [T111, T11, REF_TREE])
    nt.ok_(list(REF_TREE.children[0].children[1].iupstream()) ==
           [T112, T11, REF_TREE])


    nt.ok_(list(REF_TREE.children[1].iupstream()) == [T12, REF_TREE])
    nt.ok_(list(REF_TREE.children[1].children[0].iupstream()) ==
           [T121, T12, REF_TREE])
    nt.ok_(list(REF_TREE.children[1].children[1].iupstream()) ==
           [T122, T12, REF_TREE])


def test_leaf_iteration():
    ileaf = Tree.ileaf
    nt.ok_(list(ileaf(REF_TREE)) == [T111, T112, T12111, T12112, T122])
    nt.ok_(list(ileaf(REF_TREE.children[0])) == [T111, T112])
    nt.ok_(list(ileaf(REF_TREE.children[1])) == [T12111, T12112, T122])
    nt.ok_(list(ileaf(REF_TREE.children[0].children[0])) == [T111])
    nt.ok_(list(ileaf(REF_TREE.children[0].children[1])) == [T112])
    nt.ok_(list(ileaf(REF_TREE.children[1].children[0])) == [T12111, T12112])
    nt.ok_(list(ileaf(REF_TREE.children[1].children[1])) == [T122])


def test_iforking_point():
    nt.assert_equal([n for n in REF_TREE2.iforking_point()],
                    [REF_TREE2, T11_, T1111_, T12_, T1211_])


def test_iforking_point_postorder():
    nt.assert_equal([n for n in REF_TREE2.iforking_point(Tree.ipostorder)],
                    [T1111_, T11_, T1211_, T12_, REF_TREE2])


def test_iforking_point_upstream():
    ileaf = Tree.ileaf
    leaves = [l for l in ileaf(REF_TREE2)]
    ref_paths = [
        [T1111_, T11_, REF_TREE2], [T1111_, T11_, REF_TREE2], [T1111_, T11_, REF_TREE2],
        [T11_, REF_TREE2], [T1211_, T12_, REF_TREE2], [T1211_, T12_, REF_TREE2],
        [T12_, REF_TREE2]
    ]

    for l, ref in zip(leaves, ref_paths):
        nt.assert_equal([s for s in l.iforking_point(Tree.iupstream)], ref)


def test_ibifurcation_point():
    nt.assert_equal([n for n in REF_TREE2.ibifurcation_point()],
                    [REF_TREE2, T11_, T12_, T1211_])


def test_ibifurcation_point_postorder():
    nt.assert_equal([n for n in REF_TREE2.ibifurcation_point(Tree.ipostorder)],
                    [T11_, T1211_, T12_, REF_TREE2])


def test_ibifurcation_point_upstream():
    leaves = [l for l in REF_TREE2.ileaf()]
    ref_paths = [
        [T11_, REF_TREE2], [T11_, REF_TREE2], [T11_, REF_TREE2], [T11_, REF_TREE2],
        [T1211_, T12_, REF_TREE2], [T1211_, T12_, REF_TREE2], [T12_, REF_TREE2]
    ]

    for l, ref in zip(leaves, ref_paths):
        nt.assert_equal([s for s in l.ibifurcation_point(Tree.iupstream)], ref)


def test_valiter_bifurcation_point():
    nt.ok_(list(REF_TREE2.ibifurcation_point()) ==
           [REF_TREE2, T11_, T12_, T1211_])
