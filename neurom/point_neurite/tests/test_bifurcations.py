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
from neurom.point_neurite import io
from neurom.point_neurite.io.utils import make_neuron
from neurom.point_neurite.point_tree import PointTree
from neurom.point_neurite import bifurcations as bif
from neurom import iter_neurites
import math


class MockNeuron(object):
    pass


DATA_PATH = './test_data'
SWC_PATH = os.path.join(DATA_PATH, 'swc/')

data    = io.load_data(SWC_PATH + 'Neuron.swc')
neuron0 = make_neuron(data)
tree0   = neuron0.neurites[0]

def _make_neuron_tree():
    p = [0.0, 0.0, 0.0, 1.0, 1, 1, 2]
    T = PointTree(p)
    T1 = T.add_child(PointTree([0.0, 1.0, 0.0, 1.0, 1, 1, 2]))
    T2 = T1.add_child(PointTree([0.0, 2.0, 0.0, 1.0, 1, 1, 2]))
    T3 = T2.add_child(PointTree([0.0, 4.0, 0.0, 2.0, 1, 1, 2]))
    T4 = T3.add_child(PointTree([0.0, 5.0, 0.0, 2.0, 1, 1, 2]))
    T5 = T4.add_child(PointTree([2.0, 5.0, 0.0, 1.0, 1, 1, 2]))
    T6 = T4.add_child(PointTree([0.0, 5.0, 2.0, 1.0, 1, 1, 2]))
    T7 = T5.add_child(PointTree([3.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T8 = T7.add_child(PointTree([4.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T9 = T6.add_child(PointTree([0.0, 5.0, 3.0, 0.75, 1, 1, 2]))
    T10 = T9.add_child(PointTree([0.0, 6.0, 3.0, 0.75, 1, 1, 2]))
    return T


def _make_simple_tree():
    p = [0.0, 0.0, 0.0, 1.0, 1, 1, 1]
    T = PointTree(p)
    T1 = T.add_child(PointTree([0.0, 2.0, 0.0, 1.0, 1, 1, 1]))
    T2 = T1.add_child(PointTree([0.0, 4.0, 0.0, 1.0, 1, 1, 1]))
    T3 = T2.add_child(PointTree([0.0, 6.0, 0.0, 1.0, 1, 1, 1]))
    T4 = T3.add_child(PointTree([0.0, 8.0, 0.0, 1.0, 1, 1, 1]))

    T5 = T.add_child(PointTree([0.0, 0.0, 2.0, 1.0, 1, 1, 1]))
    T6 = T5.add_child(PointTree([0.0, 0.0, 4.0, 1.0, 1, 1, 1]))
    T7 = T6.add_child(PointTree([0.0, 0.0, 6.0, 1.0, 1, 1, 1]))
    T8 = T7.add_child(PointTree([0.0, 0.0, 8.0, 1.0, 1, 1, 1]))

    return T


NEURON_TREE = _make_neuron_tree()
SIMPLE_TREE = _make_simple_tree()

NEURON = MockNeuron()
NEURON.neurites = [NEURON_TREE]

SIMPLE_NEURON = MockNeuron()
SIMPLE_NEURON.neurites = [SIMPLE_TREE]

def _make_branching_tree():
    '''This tree has 3 branching points'''
    p = [0.0, 0.0, 0.0, 1.0, 1, 1, 2]
    T = PointTree(p)
    T1 = T.add_child(PointTree([0.0, 1.0, 0.0, 1.0, 1, 1, 2]))
    T2 = T1.add_child(PointTree([0.0, 2.0, 0.0, 1.0, 1, 1, 2]))
    T3 = T2.add_child(PointTree([0.0, 4.0, 0.0, 2.0, 1, 1, 2]))
    T4 = T3.add_child(PointTree([0.0, 5.0, 0.0, 2.0, 1, 1, 2]))
    T5 = T4.add_child(PointTree([2.0, 5.0, 0.0, 1.0, 1, 1, 2]))
    T6 = T4.add_child(PointTree([0.0, 5.0, 2.0, 1.0, 1, 1, 2]))
    T7 = T5.add_child(PointTree([3.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T8 = T7.add_child(PointTree([4.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T9 = T6.add_child(PointTree([0.0, 5.0, 3.0, 0.75, 1, 1, 2]))
    T10 = T9.add_child(PointTree([0.0, 6.0, 3.0, 0.75, 1, 1, 2]))
    T11 = T9.add_child(PointTree([0.0, 6.0, 4.0, 0.75, 1, 1, 2]))
    T33 = T3.add_child(PointTree([1.0, 5.0, 0.0, 2.0, 1, 1, 2]))
    T331 = T33.add_child(PointTree([15.0, 15.0, 0.0, 2.0, 1, 1, 2]))
    return T


def _make_branching_tree_zero_length_bif_segments():
    '''This tree has 3 branching points'''
    p = [0.0, 0.0, 0.0, 1.0, 1, 1, 2]
    T = PointTree(p)
    T1 = T.add_child(PointTree([0.0, 1.0, 0.0, 1.0, 1, 1, 2]))
    T2 = T1.add_child(PointTree([0.0, 2.0, 0.0, 1.0, 1, 1, 2]))
    T3 = T2.add_child(PointTree([0.0, 4.0, 0.0, 2.0, 1, 1, 2]))
    T4 = T3.add_child(PointTree([0.0, 5.0, 0.0, 2.0, 1, 1, 2]))
    T5_0 = T4.add_child(PointTree([0.0, 5.0, 0.0, 2.0, 1, 1, 2]))
    T5 = T5_0.add_child(PointTree([2.0, 5.0, 0.0, 1.0, 1, 1, 2]))
    T6_0 = T4.add_child(PointTree([0.0, 5.0, 0.0, 2.0, 1, 1, 2]))
    T6 = T6_0.add_child(PointTree([0.0, 5.0, 2.0, 1.0, 1, 1, 2]))
    T7 = T5.add_child(PointTree([3.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T8 = T7.add_child(PointTree([4.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T9 = T6.add_child(PointTree([0.0, 5.0, 3.0, 0.75, 1, 1, 2]))
    T10_0 = T9.add_child(PointTree([0.0, 5.0, 3.0, 0.75, 1, 1, 2]))
    T10 = T10_0.add_child(PointTree([0.0, 6.0, 3.0, 0.75, 1, 1, 2]))
    T11 = T9.add_child(PointTree([0.0, 6.0, 4.0, 0.75, 1, 1, 2]))
    T33 = T3.add_child(PointTree([1.0, 5.0, 0.0, 2.0, 1, 1, 2]))
    T331 = T33.add_child(PointTree([15.0, 15.0, 0.0, 2.0, 1, 1, 2]))
    return T


BRANCHING_TREE = _make_branching_tree()
BRANCHING_NEURON = MockNeuron()
BRANCHING_NEURON.neurites = [BRANCHING_TREE]

ZERO_SEG_BIF_TREE = _make_branching_tree_zero_length_bif_segments()

def _make_odd_tree():
    ''' The infamous invisible tree. This should fail check
    in case of integer division in partition
    '''
    p = [0.0, 0.0, 0.0, 0.0, 0, 0, 0]
    T = PointTree(p)
    T.add_child(PointTree([0.0, 0.0, 0.0, 0.0, 0, 0, 0]))
    T.add_child(PointTree([0.0, 0.0, 0.0, 0.0, 0, 0, 0]))
    T.children[0].add_child(PointTree([0.0, 0.0, 0.0, 0.0, 0, 0, 0]))
    T.children[0].add_child(PointTree([0.0, 0.0, 0.0, 0.0, 0, 0, 0]))
    T.children[1].add_child(PointTree([0.0, 0.0, 0.0, 0.0, 0, 0, 0]))
    T.children[1].add_child(PointTree([0.0, 0.0, 0.0, 0.0, 0, 0, 0]))
    T.children[1].children[0].add_child(PointTree([0.0, 0.0, 0.0, 0.0, 0, 0, 0]))
    T.children[1].children[0].add_child(PointTree([0.0, 0.0, 0.0, 0.0, 0, 0, 0]))
    return T


ODD_TREE = _make_odd_tree()
ODD_NEURON = MockNeuron()
ODD_NEURON.neurites = [ODD_TREE]

def _check_local_bifurcation_angles(obj):

    angles = [a for a in iter_neurites(obj, bif.local_angle)]

    nt.eq_(angles, [math.pi / 4, math.pi / 2, math.pi / 4])


def _check_remote_bifurcation_angles(obj):

    angles = [a for a in iter_neurites(obj, bif.remote_angle)]

    nt.eq_(angles,
           [0.9380474917927135, math.pi / 2, math.pi / 4])


def _check_partition(obj):
    p = [a for a in iter_neurites(ODD_NEURON, bif.partition)]

    nt.eq_(p, [5./3., 1., 3., 1.])


def _check_count(obj, n):
    nt.eq_(bif.count(obj), n)


def _check_points(obj):
    @bif.bifurcation_point_function(as_tree=False)
    def point(bif):
        return bif[:4]

    bif_points = [p for p in iter_neurites(obj, point)]
    nt.eq_(bif_points,
           [[0.0, 4.0, 0.0, 2.0], [0.0, 5.0, 0.0, 2.0], [0.0, 5.0, 3.0, 0.75]])


def test_local_bifurcation_angle():
    _check_local_bifurcation_angles(BRANCHING_NEURON)
    _check_local_bifurcation_angles(BRANCHING_TREE)
    _check_local_bifurcation_angles(ZERO_SEG_BIF_TREE)


def test_remote_bifurcation_angle():
    _check_remote_bifurcation_angles(BRANCHING_NEURON)
    _check_remote_bifurcation_angles(BRANCHING_TREE)


def test_partition():
    _check_partition(ODD_TREE)
    _check_partition(ODD_NEURON)


def test_points():
    _check_points(BRANCHING_NEURON)


def test_count():
    _check_count(BRANCHING_TREE, 3)
    _check_count(BRANCHING_NEURON, 3)

    neuron_b = MockNeuron()
    neuron_b.neurites = [BRANCHING_TREE, BRANCHING_TREE, BRANCHING_TREE]

    _check_count(neuron_b, 9)
