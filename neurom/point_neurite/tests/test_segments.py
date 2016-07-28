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
from neurom.point_neurite import segments as seg
from neurom import iter_neurites

import math
from itertools import izip


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

BRANCHING_TREE = _make_branching_tree()




def _check_segment_lengths(obj):

    lg = [l for l in iter_neurites(obj, seg.length)]

    nt.eq_(lg, [1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0])


def _check_segment_lengths2(obj):

    lg = [l for l in iter_neurites(obj, seg.length2)]

    nt.eq_(lg, [1.0, 1.0, 4.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0])


def _check_segment_volumes(obj):

    sv = (l/math.pi for l in iter_neurites(obj, seg.volume))

    ref = (1.0, 1.0, 4.6666667, 4.0, 4.6666667, 0.7708333,
           0.5625, 4.6666667, 0.7708333, 0.5625)

    for a, b in izip(sv, ref):
        nt.assert_almost_equal(a, b)


def _check_segment_areas(obj):

    sa = (l/math.pi for l in iter_neurites(obj, seg.area))

    ref = (2.0, 2.0, 6.7082039, 4.0, 6.7082039, 1.8038587,
           1.5, 6.7082039, 1.8038587, 1.5)

    for a, b in izip(sa, ref):
        nt.assert_almost_equal(a, b)


def _check_segment_radius(obj):

    rad = [r for r in iter_neurites(obj, seg.radius)]

    nt.eq_(rad,
           [1.0, 1.0, 1.5, 2.0, 1.5, 0.875, 0.75, 1.5, 0.875, 0.75])


def _check_segment_x_coordinate(obj):

    xcoord = [s for s in iter_neurites(obj, seg.x_coordinate)]

    nt.eq_(xcoord,
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _check_segment_y_coordinate(obj):

    ycoord = [s for s in iter_neurites(obj, seg.y_coordinate)]

    nt.eq_(ycoord,
           [1.0, 3.0, 5.0, 7.0, 0.0, 0.0, 0.0, 0.0])


def _check_segment_z_coordinate(obj):

    zcoord = [s for s in iter_neurites(obj, seg.z_coordinate)]

    nt.eq_(zcoord,
           [0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 7.0])


def _check_count(obj, n):
    nt.eq_(seg.count(obj), n)


def _check_segment_radial_dists(obj):

    origin = [0.0, 0.0, 0.0]

    rd = [d for d in iter_neurites(SIMPLE_NEURON, seg.radial_dist(origin))]

    nt.eq_(rd, [1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0])


def _check_segment_taper_rate(obj):

    tp = [t for t in iter_neurites(obj, seg.taper_rate)]

    nt.eq_(tp,
           [0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0])


def test_segment_volumes():
    _check_segment_volumes(NEURON)
    _check_segment_volumes(NEURON_TREE)


def test_segment_lengths():
    _check_segment_lengths(NEURON)
    _check_segment_lengths(NEURON_TREE)


def test_segment_lengths2():
    _check_segment_lengths2(NEURON)
    _check_segment_lengths2(NEURON_TREE)


def test_segment_areas():
    _check_segment_areas(NEURON)
    _check_segment_areas(NEURON_TREE)


def test_segment_radius():
    _check_segment_radius(NEURON)
    _check_segment_radius(NEURON_TREE)

def test_segment_x_coordinate():
    _check_segment_x_coordinate(SIMPLE_TREE)

def test_segment_y_coordinate():
    _check_segment_y_coordinate(SIMPLE_TREE)

def test_segment_z_coordinate():
    _check_segment_z_coordinate(SIMPLE_TREE)

def test_count():
    _check_count(NEURON, 10)
    _check_count(NEURON_TREE, 10)

    neuron_b = MockNeuron()
    neuron_b.neurites = [NEURON_TREE, NEURON_TREE, NEURON_TREE]

    _check_count(neuron_b, 30)


def test_segment_radial_dists():
    _check_segment_radial_dists(SIMPLE_NEURON)
    _check_segment_radial_dists(SIMPLE_TREE)


def test_segment_taper_rate():
    _check_segment_taper_rate(NEURON)
    _check_segment_taper_rate(NEURON_TREE)


def test_cross_section_at_fraction():

    res = seg.cross_section_at_fraction((PointTree((1.,1.,1., 1.)),PointTree((2.,2.,2., 2.))), 0.5)
    print res
    nt.eq_(tuple(res[0]), (1.5, 1.5, 1.5))
    nt.assert_equal(res[1], 1.5)
