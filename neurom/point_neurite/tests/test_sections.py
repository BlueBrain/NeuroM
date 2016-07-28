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
from neurom import iter_neurites
from neurom.point_neurite import io
from neurom.point_neurite.io.utils import make_neuron
from neurom.point_neurite.point_tree import PointTree, isection
from neurom.point_neurite import segments as seg
from neurom.point_neurite import sections as sec


# Mock tree holding integers, not points
MOCK_TREE = PointTree(0)
MOCK_TREE.add_child(PointTree(11))
MOCK_TREE.add_child(PointTree(12))
MOCK_TREE.children[0].add_child(PointTree(111))
MOCK_TREE.children[0].add_child(PointTree(112))
MOCK_TREE.children[1].add_child(PointTree(121))
MOCK_TREE.children[1].add_child(PointTree(122))
MOCK_TREE.children[1].children[0].add_child(PointTree(1211))
MOCK_TREE.children[1].children[0].children[0].add_child(PointTree(12111))
MOCK_TREE.children[1].children[0].children[0].add_child(PointTree(12112))


class MockNeuron(object):
    pass


DATA_PATH = './test_data'
SWC_PATH = os.path.join(DATA_PATH, 'swc/')

data    = io.load_data(SWC_PATH + 'Neuron.swc')
neuron0 = make_neuron(data)
tree0   = neuron0.neurites[0]


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


SIMPLE_TREE = _make_simple_tree()
SIMPLE_NEURON = MockNeuron()
SIMPLE_NEURON.neurites = [SIMPLE_TREE]

NEURON_TREE = _make_branching_tree()

NEURON = MockNeuron()
NEURON.neurites = [NEURON_TREE]

def _check_count(obj, n):
    nt.eq_(sec.count(obj), n)


def _check_length(obj):
    sec_len = [l for l in iter_neurites(obj, sec.length)]
    seg_len = [l for l in iter_neurites(obj, seg.length)]
    sum_sec_len = sum(sec_len)
    sum_seg_len = sum(seg_len)

    # check that sum of section lengths is same as sum of segment lengths
    nt.eq_(sum_sec_len, sum_seg_len)

    nt.assert_almost_equal(sum_sec_len, 33.0330776588)


def _check_volume(obj):
    sec_vol = [l for l in iter_neurites(obj, sec.volume)]
    seg_vol = [l for l in iter_neurites(obj, seg.volume)]
    sum_sec_vol = sum(sec_vol)
    sum_seg_vol = sum(seg_vol)

    # check that sum of section volumes is same as sum of segment lengths
    nt.assert_almost_equal(sum_sec_vol, sum_seg_vol)

    nt.assert_almost_equal(sum_sec_vol, 307.68010178856395)


def _check_area(obj):
    sec_area = [l for l in iter_neurites(obj, sec.area)]
    seg_area = [l for l in iter_neurites(obj, seg.area)]
    sum_sec_area = sum(sec_area)
    sum_seg_area = sum(seg_area)

    # check that sum of section areas is same as sum of segment lengths
    nt.assert_almost_equal(sum_sec_area, sum_seg_area)

    nt.assert_almost_equal(sum_sec_area, 349.75070138106133)


def test_count():
    _check_count(NEURON, 7)
    _check_count(NEURON_TREE, 7)

    neuron_b = MockNeuron()
    neuron_b.neurites = [NEURON_TREE, NEURON_TREE, NEURON_TREE]

    _check_count(neuron_b, 21)


def _check_section_radial_dists_end_point(obj):

    origin = [0.0, 0.0, 0.0]

    rd = [d for d in iter_neurites(obj, sec.radial_dist(origin))]

    nt.eq_(rd, [8.0, 8.0])


def _check_section_radial_dists_start_point(obj):

    origin = [0.0, 0.0, 0.0]

    rd = [d for d in iter_neurites(obj, sec.radial_dist(origin, True))]

    nt.eq_(rd, [0.0, 0.0])


def _check_path_length_end_point(obj, ref):
    pl = [l for l in iter_neurites(obj, sec.end_point_path_length)]
    nt.eq_(pl, ref)


def _check_path_length_start_point(obj, ref):
    pl = [l for l in iter_neurites(obj, sec.start_point_path_length)]
    nt.eq_(pl, ref)

def test_length():
    _check_length(NEURON)
    _check_length(NEURON_TREE)


def test_section_volume():
    _check_volume(NEURON)
    _check_volume(NEURON_TREE)


def test_section_area():
    _check_area(NEURON)
    _check_area(NEURON_TREE)


def test_section_radial_dists_end_point():
    _check_section_radial_dists_end_point(SIMPLE_NEURON)
    _check_section_radial_dists_end_point(SIMPLE_TREE)


def test_section_radial_dists_start_point():
    _check_section_radial_dists_start_point(SIMPLE_NEURON)
    _check_section_radial_dists_start_point(SIMPLE_TREE)


def test_end_point_path_length():
    simple_ref = [8, 8]
    _check_path_length_end_point(SIMPLE_NEURON, simple_ref)
    _check_path_length_end_point(SIMPLE_TREE, simple_ref)

    ref = [4.0, 5.0, 9.0, 8.0, 9.0, 9.4142135623730958, 22.618864096458349]
    _check_path_length_end_point(NEURON, ref)
    _check_path_length_end_point(NEURON_TREE, ref)


def test_start_point_path_length():
    simple_ref = [0, 0]
    _check_path_length_start_point(SIMPLE_NEURON, simple_ref)
    _check_path_length_start_point(SIMPLE_TREE, simple_ref)

    ref = [0, 4.0, 5.0, 5.0, 8.0, 8.0, 4.0]
    _check_path_length_start_point(NEURON, ref)
    _check_path_length_start_point(NEURON_TREE, ref)


def test_section_branch_order():

    sec_bo = [bo for bo in iter_neurites(MOCK_TREE, sec.branch_order)]
    nt.eq_(sec_bo, [1, 2, 2, 1, 2, 3, 3, 2])


def test_point_at_path_fraction():
    section = isection(SIMPLE_TREE).next()
    res = sec.point_at_path_fraction(section, 0.5)
    nt.eq_(tuple(res), (0., 4., 0.))
