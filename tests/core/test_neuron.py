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

from copy import copy, deepcopy
from pathlib import Path

import neurom as nm
import numpy as np
import morphio
from neurom.core.morphology import Morphology, graft_morphology, iter_segments
from numpy.testing import assert_array_equal

SWC_PATH = Path(__file__).parent.parent / 'data/swc/'


def test_simple():
    nm.load_morphology(str(SWC_PATH / 'simple.swc'))


def test_load_morphology_pathlib():
    nm.load_morphology(SWC_PATH / 'simple.swc')


def test_load_morphology_from_other_morphologies():
    filename = SWC_PATH / 'simple.swc'

    expected_points = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 5.0, 0.0, 1.0],
        [0.0, 5.0, 0.0, 1.0],
        [-5.0, 5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 1.0],
        [6.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, -4.0, 0.0, 1.0],
        [0.0, -4.0, 0.0, 1.0],
        [6.0, -4.0, 0.0, 0.0],
        [0.0, -4.0, 0.0, 1.0],
        [-5.0, -4.0, 0.0, 0.0],
    ]

    assert_array_equal(nm.load_morphology(nm.load_morphology(filename)).points, expected_points)
    assert_array_equal(nm.load_morphology(morphio.Morphology(filename)).points, expected_points)


def test_for_morphio():
    Morphology(morphio.mut.Morphology())

    morphio_m = morphio.mut.Morphology()
    morphio_m.soma.points = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    morphio_m.soma.diameters = [1, 1, 1]

    neurom_m = Morphology(morphio_m)
    assert_array_equal(
        neurom_m.soma.points, [[0.0, 0.0, 0.0, 0.5], [1.0, 1.0, 1.0, 0.5], [2.0, 2.0, 2.0, 0.5]]
    )


def _check_cloned_morphology(m, m2):
    # check if two morphs are identical

    # soma
    assert isinstance(m2.soma, type(m.soma))
    assert m.soma.radius == m2.soma.radius

    for v1, v2 in zip(m.soma.iter(), m2.soma.iter()):
        assert np.allclose(v1, v2)

    # neurites
    for v1, v2 in zip(iter_segments(m), iter_segments(m2)):
        (v1_start, v1_end), (v2_start, v2_end) = v1, v2
        assert np.allclose(v1_start, v2_start)
        assert np.allclose(v1_end, v2_end)

    # check if the ids are different
    # somata
    assert m.soma is not m2.soma

    # neurites
    for neu1, neu2 in zip(m.neurites, m2.neurites):
        assert neu1 is not neu2


def test_copy():
    m = nm.load_morphology(SWC_PATH / 'simple.swc')
    _check_cloned_morphology(m, copy(m))


def test_deepcopy():
    m = nm.load_morphology(SWC_PATH / 'simple.swc')
    _check_cloned_morphology(m, deepcopy(m))


def test_eq():
    m1 = nm.load_morphology(SWC_PATH / 'simple.swc').neurites[1]
    m2 = nm.load_morphology(SWC_PATH / 'simple.swc').neurites[1]
    assert m1 == m2

    m1.process_subtrees = True
    assert m1 != m2


def test_graft_morphology():
    m = nm.load_morphology(SWC_PATH / 'simple.swc')
    basal_dendrite = m.neurites[0]
    m2 = graft_morphology(basal_dendrite.root_node)
    assert len(m2.neurites) == 1
    assert basal_dendrite == m2.neurites[0]


def test_str():
    n = nm.load_morphology(SWC_PATH / 'simple.swc')
    assert 'Morphology' in str(n)
    assert 'Section' in str(n.neurites[0].root_node)
