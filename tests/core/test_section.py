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

from pathlib import Path
import neurom as nm
from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np

SWC_PATH = Path(__file__).parent.parent / 'data/swc/'


def test_section_base_func():
    nrn = nm.load_neuron(str(SWC_PATH / 'simple.swc'))
    section = nrn.sections[0]

    assert section.type == nm.NeuriteType.basal_dendrite
    assert section.id == 0
    assert_array_equal(section.points, [[0, 0, 0, 1], [0, 5, 0, 1]])
    assert_almost_equal(section.length, 5)
    assert_almost_equal(section.area, 31.41592653589793)
    assert_almost_equal(section.volume, 15.707963267948964)


def test_section_tree():
    nrn = nm.load_neuron(str(SWC_PATH / 'simple.swc'))

    assert nrn.sections[0].parent is None
    assert nrn.sections[0] == nrn.sections[0].children[0].parent

    assert_array_equal([s.is_root() for s in nrn.sections],
                       [True, False, False, True, False, False])
    assert_array_equal([s.is_leaf() for s in nrn.sections],
                       [False, True, True, False, True, True])
    assert_array_equal([s.is_forking_point() for s in nrn.sections],
                       [True, False, False, True, False, False])
    assert_array_equal([s.is_bifurcation_point() for s in nrn.sections],
                       [True, False, False, True, False, False])
    assert_array_equal([s.id for s in nrn.neurites[0].root_node.ipreorder()],
                       [0, 1, 2])
    assert_array_equal([s.id for s in nrn.neurites[0].root_node.ipostorder()],
                       [1, 2, 0])
    assert_array_equal([s.id for s in nrn.neurites[0].root_node.iupstream()],
                       [0])
    assert_array_equal([s.id for s in nrn.sections[2].iupstream()],
                       [2, 0])
    assert_array_equal([s.id for s in nrn.neurites[0].root_node.ileaf()],
                       [1, 2])
    assert_array_equal([s.id for s in nrn.sections[2].ileaf()],
                       [2])
    assert_array_equal([s.id for s in nrn.neurites[0].root_node.iforking_point()],
                       [0])
    assert_array_equal([s.id for s in nrn.neurites[0].root_node.ibifurcation_point()],
                       [0])


def test_append_section():
    n = nm.load_neuron(SWC_PATH / 'simple.swc')
    s = n.sections[0]

    s.append_section(n.sections[-1])
    assert len(s.children) == 3
    assert s.children[-1].id == 6
    assert s.children[-1].type == n.sections[-1].type

    s.append_section(n.sections[-1].morphio_section)
    assert len(s.children) == 4
    assert s.children[-1].id == 7
    assert s.children[-1].type == n.sections[-1].type


def test_set_points():
    n = nm.load_neuron(SWC_PATH / 'simple.swc')
    s = n.sections[0]
    s.points = np.array([
        [0, 5, 0, 2],
        [0, 7, 0, 2],
    ])
    assert_array_equal(s.points, np.array([
        [0, 5, 0, 2],
        [0, 7, 0, 2],
    ]))
