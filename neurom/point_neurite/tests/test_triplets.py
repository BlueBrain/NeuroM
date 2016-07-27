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
from neurom.point_neurite import triplets as trip
from neurom import iter_neurites

import math


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
    T2 = T1.add_child(PointTree([2.0, 2.0, 0.0, 1.0, 1, 1, 1]))
    T3 = T2.add_child(PointTree([2.0, 6.0, 0.0, 1.0, 1, 1, 1]))

    T5 = T.add_child(PointTree([0.0, 0.0, 2.0, 1.0, 1, 1, 1]))
    T6 = T5.add_child(PointTree([2.0, 0.0, 2.0, 1.0, 1, 1, 1]))
    T7 = T6.add_child(PointTree([6.0, 0.0, 2.0, 1.0, 1, 1, 1]))

    return T


SIMPLE_TREE = _make_simple_tree()
SIMPLE_NEURON = MockNeuron()
SIMPLE_NEURON.neurites = [SIMPLE_TREE]


def _check_meander_angles(obj):

    angles = [a for a in iter_neurites(obj, trip.meander_angle)]

    nt.eq_(angles,
           [math.pi / 2, math.pi / 2, math.pi / 2, math.pi])


def _check_count(obj, n):
    nt.eq_(trip.count(obj), n)


def test_meander_angles():
    _check_meander_angles(SIMPLE_TREE)
    _check_meander_angles(SIMPLE_NEURON)


def test_count():
    _check_count(SIMPLE_NEURON, 4)
    _check_count(SIMPLE_TREE, 4)

    neuron_b = MockNeuron()
    neuron_b.neurites = [SIMPLE_TREE, SIMPLE_TREE, SIMPLE_TREE]

    _check_count(neuron_b, 12)
