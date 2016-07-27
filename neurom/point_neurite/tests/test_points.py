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
from neurom import io
from neurom.point_neurite.point_tree import PointTree
from neurom.point_neurite.io.utils import make_neuron
from neurom.point_neurite import points as pts
from neurom import iter_neurites


class MockNeuron(object):
    pass


def _make_tree():
    '''This tree has 3 branching points'''
    p = [0.0, 0.0, 0.0, 1.0, 1, 1, 2]
    T = PointTree(p)
    T1 = T.add_child(PointTree([0.0, 1.0, 0.0, 2.0, 1, 1, 2]))
    T2 = T1.add_child(PointTree([0.0, 2.0, 0.0, 3.0, 1, 1, 2]))
    T3 = T2.add_child(PointTree([0.0, 4.0, 0.0, 4.0, 1, 1, 2]))
    T4 = T3.add_child(PointTree([0.0, 5.0, 0.0, 5.0, 1, 1, 2]))
    T5 = T4.add_child(PointTree([2.0, 5.0, 0.0, 6.0, 1, 1, 2]))
    T6 = T4.add_child(PointTree([0.0, 5.0, 2.0, 7.0, 1, 1, 2]))
    return T

TREE = _make_tree()
TREES = [TREE, TREE, TREE]
NEURON = MockNeuron()
NEURON.neurites = [TREE]
POPULATION = MockNeuron()
POPULATION.neurites = [TREE, TREE, TREE]

def _check_radius0(obj):

    radii = [r for r in iter_neurites(obj, pts.radius)]
    nt.eq_(radii, [1, 2, 3, 4, 5, 6, 7])


def _check_radius1(obj):

    radii = [r for r in iter_neurites(obj, pts.radius)]
    nt.eq_(radii, [1, 2, 3, 4, 5, 6, 7,
                   1, 2, 3, 4, 5, 6, 7,
                   1, 2, 3, 4, 5, 6, 7])


def _check_count(obj, n):
    nt.eq_(pts.count(obj), n)


def test_radius():
    _check_radius0(NEURON)
    _check_radius0(TREE)
    _check_radius1(TREES)
    _check_radius1(POPULATION)


def test_count():
    _check_count(TREE, 7)
    _check_count(NEURON, 7)
    _check_count(POPULATION, 21)
    _check_count(TREES, 21)
