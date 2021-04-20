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

from copy import deepcopy
from io import StringIO
from pathlib import Path

import numpy as np
from neurom import load_neuron
from neurom.check import morphtree as mt
from neurom.core.dataformat import COLS

DATA_PATH = Path(__file__).parent.parent / 'data'
SWC_PATH = DATA_PATH / 'swc'


def _make_flat(neuron):

    class Flattenizer:
        def __call__(self, points):
            points = deepcopy(points)
            points[:, COLS.Z] = 0.
            return points

    return neuron.transform(Flattenizer())


def _make_monotonic(neuron):
    for neurite in neuron.neurites:
        for section in neurite.iter_sections():
            points = section.points
            if section.parent is not None:
                points[0][COLS.R] = section.parent.points[-1][COLS.R] / 2
            for point_id in range(len(points) - 1):
                points[point_id + 1][COLS.R] = points[point_id][COLS.R] / 2.
            section.points = points


def _generate_back_track_tree(n, dev):
    points = np.array(dev) + np.array([1, 3 if n == 0 else -3, 0])

    neuron = load_neuron(StringIO(u"""
    ((CellBody)
     (0 0 0 0.4))

    ((Dendrite)
    (0 0 0 0.4)
    (0 1 0 0.3)
    (0 2 0 0.28)
    (
      (0 2 0 0.28)
      (1 3 0 0.3)
      (2 4 0 0.22)
      |
      (0 2 0 0.28)
      (1 -3 0 0.3)
      (2 -4 0 0.24)
      ({0} {1} {2} 0.52)
      (3 -5 0 0.2)
      (4 -6 0 0.2)
    ))
    """.format(*points.tolist())), reader='asc')

    return neuron


def test_is_monotonic():
    # tree with decreasing radii
    neuron = load_neuron(StringIO(u"""
        ((Dendrite)
        (0 0 0 1.0)
        (0 0 0 0.99)
        (
          (0 0 0 0.99)
          (0 0 0 0.1)
          |
          (0 0 0 0.5)
          (0 0 0 0.2)
        ))"""), reader='asc')
    assert mt.is_monotonic(neuron.neurites[0], 1e-6)

    # tree with equal radii
    neuron = load_neuron(StringIO(u"""
        ((Dendrite)
        (0 0 0 1.0)
        (0 0 0 1.0)
        (
          (0 0 0 1.0)
          (0 0 0 1.0)
          |
          (0 0 0 1.0)
          (0 0 0 1.0)
        ))"""), reader='asc')
    assert mt.is_monotonic(neuron.neurites[0], 1e-6)

    # tree with increasing radii
    neuron = load_neuron(StringIO(u"""
        ((Dendrite)
        (0 0 0 1.0)
        (0 0 0 1.0)
        (
          (0 0 0 1.1)
          (0 0 0 1.1)
          |
          (0 0 0 0.3)
          (0 0 0 0.1)
        ))"""), reader='asc')
    assert not mt.is_monotonic(neuron.neurites[0], 1e-6)

    # Tree with larger child initial point
    neuron = load_neuron(StringIO(u"""
        ((Dendrite)
        (0 0 0 1.0)
        (0 0 0 0.75)
        (0 0 0 0.5)
        (0 0 0 0.25)
        (
          (0 0 0 0.375)
          (0 0 0 0.125)
          (0 0 0 0.625)
        ))"""), reader='asc')
    assert not mt.is_monotonic(neuron.neurites[0], 1e-6)


def test_is_flat():
    neu_tree = load_neuron(Path(SWC_PATH, 'Neuron.swc'))
    assert not mt.is_flat(neu_tree.neurites[0], 1e-6, method='tolerance')
    assert not mt.is_flat(neu_tree.neurites[0], 0.1, method='ratio')


def test_is_back_tracking():
    # case 1: a back-track falls directly on a previous node
    t = _generate_back_track_tree(1, (0., 0., 0.))
    assert mt.is_back_tracking(t.neurites[0])

    # case 2: a zigzag is close to another segment
    t = _generate_back_track_tree(1, (0.1, -0.1, 0.02))
    assert mt.is_back_tracking(t.neurites[0])

    # case 3: a zigzag is close to another segment 2
    t = _generate_back_track_tree(1, (-0.2, 0.04, 0.144))
    assert mt.is_back_tracking(t.neurites[0])

    # case 4: a zigzag far from civilization
    t = _generate_back_track_tree(1, (10., -10., 10.))
    assert not mt.is_back_tracking(t.neurites[0])

    # case 5: a zigzag on another section
    # currently zigzag is defined on the same section
    # thus this test should not be true
    t = _generate_back_track_tree(0, (-0.2, 0.04, 0.144))
    assert not mt.is_back_tracking(t.neurites[0])


def test_get_flat_neurites():
    n = load_neuron(Path(SWC_PATH, 'Neuron.swc'))
    assert len(mt.get_flat_neurites(n, 1e-6, method='tolerance')) == 0
    assert len(mt.get_flat_neurites(n, 0.1, method='ratio')) == 0

    n = _make_flat(n)
    assert len(mt.get_flat_neurites(n, 1e-6, method='tolerance')) == 4
    assert len(mt.get_flat_neurites(n, 0.1, method='ratio')) == 4


def test_get_nonmonotonic_neurites():
    n = load_neuron(Path(SWC_PATH, 'Neuron.swc'))
    assert len(mt.get_nonmonotonic_neurites(n)) == 4
    _make_monotonic(n)
    assert len(mt.get_nonmonotonic_neurites(n)) == 0


def test_get_back_tracking_neurites():
    n = load_neuron(Path(SWC_PATH, 'Neuron.swc'))
    assert len(mt.get_back_tracking_neurites(n)) == 4
