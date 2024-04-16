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

from io import StringIO
from pathlib import Path

import numpy as np
from neurom import load_morphology
from neurom.check import morphtree as mt

DATA_PATH = Path(__file__).parent.parent / 'data'
SWC_PATH = DATA_PATH / 'swc'


def _generate_back_track_tree(n, dev):
    points = np.array(dev) + np.array([1, 3 if n == 0 else -3, 0])

    m = load_morphology(StringIO(u"""
    ((CellBody) (-1 0 0 2) (1 0 0 2))

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

    return m


def test_is_monotonic():
    # tree with decreasing radii
    m = load_morphology(StringIO(u"""
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
    assert mt.is_monotonic(m.neurites[0], 1e-6)

    # tree with equal radii
    m = load_morphology(StringIO(u"""
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
    assert mt.is_monotonic(m.neurites[0], 1e-6)

    # tree with increasing radii
    m = load_morphology(StringIO(u"""
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
    assert not mt.is_monotonic(m.neurites[0], 1e-6)

    # Tree with larger child initial point
    m = load_morphology(StringIO(u"""
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
    assert not mt.is_monotonic(m.neurites[0], 1e-6)


def test_is_flat():
    m = load_morphology(Path(SWC_PATH, 'Neuron.swc'))
    assert not mt.is_flat(m.neurites[0], 1e-6, method='tolerance')
    assert not mt.is_flat(m.neurites[0], 0.1, method='ratio')


def test_back_tracking_segments():
    # case 1: a back-track falls directly on a previous node
    t1 = _generate_back_track_tree(1, (0.0, 0.0, 0.0))
    assert list(mt.back_tracking_segments(t1.neurites[0])) == [(2, 1, 0), (2, 1, 1)]

    # case 2: a zigzag is close to another segment
    t2 = _generate_back_track_tree(1, (0.1, -0.1, 0.02))
    assert list(mt.back_tracking_segments(t2.neurites[0])) == [(2, 1, 0), (2, 1, 1)]

    # case 3: a zigzag is close to another segment 2
    t3 = _generate_back_track_tree(1, (-0.2, 0.04, 0.144))
    assert list(mt.back_tracking_segments(t3.neurites[0])) == [(2, 1, 0)]

    # case 4: a zigzag far from civilization
    t4 = _generate_back_track_tree(1, (10.0, -10.0, 10.0))
    assert list(mt.back_tracking_segments(t4.neurites[0])) == []

    # case 5: a zigzag on another section
    # currently zigzag is defined on the same section
    # thus this test should not be true
    t5 = _generate_back_track_tree(0, (-0.2, 0.04, 0.144))
    assert list(mt.back_tracking_segments(t5.neurites[0])) == []


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
    m = load_morphology(Path(SWC_PATH, 'Neuron.swc'))
    assert len(mt.get_flat_neurites(m, 1e-6, method='tolerance')) == 0
    assert len(mt.get_flat_neurites(m, 0.1, method='ratio')) == 0

    m = load_morphology(Path(SWC_PATH, 'Neuron-flat.swc'))
    assert len(mt.get_flat_neurites(m, 1e-6, method='tolerance')) == 4
    assert len(mt.get_flat_neurites(m, 0.1, method='ratio')) == 4


def test_get_nonmonotonic_neurites():
    m = load_morphology(Path(SWC_PATH, 'Neuron.swc'))
    assert len(mt.get_nonmonotonic_neurites(m)) == 4

    m = load_morphology(Path(SWC_PATH, 'Neuron-monotonic.swc'))
    assert len(mt.get_nonmonotonic_neurites(m)) == 0


def test_get_back_tracking_neurites():
    m = load_morphology(Path(SWC_PATH, 'Neuron.swc'))
    assert len(mt.get_back_tracking_neurites(m)) == 4


def test_get_duplicated_point_neurites():
    m = load_morphology(Path(SWC_PATH, 'Neuron.swc'))
    assert len(mt.get_duplicated_point_neurites(m)) == 0
    assert len(mt.get_duplicated_point_neurites(m, tolerance=0.09)) == 1
    assert len(mt.get_duplicated_point_neurites(m, tolerance=999)) == 4
