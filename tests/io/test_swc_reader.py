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

import numpy as np

from morphio import RawDataError, MorphioError
from neurom import load_neuron, NeuriteType

import pytest
from numpy.testing import assert_array_equal

DATA_PATH = Path(__file__).parent.parent / 'data'
SWC_PATH = DATA_PATH / 'swc'
SWC_SOMA_PATH = SWC_PATH / 'soma'


def test_repeated_id():
    with pytest.raises(RawDataError):
        load_neuron(SWC_PATH / 'repeated_id.swc')


def test_neurite_followed_by_soma():
    with pytest.raises(MorphioError, match='Warning: found a disconnected neurite'):
        load_neuron(SWC_PATH / 'soma_with_neurite_parent.swc')


def test_read_single_neurite():
    n = load_neuron(SWC_PATH / 'point_soma_single_neurite.swc')
    assert len(n.neurites) == 1
    assert n.neurites[0].root_node.id == 0
    assert_array_equal(n.soma.points,
                       [[0, 0, 0, 3.0]])
    assert len(n.neurites) == 1
    assert len(n.sections) == 1
    assert_array_equal(n.neurites[0].points,
                       np.array([[0, 0, 2, 0.5],
                                 [0, 0, 3, 0.5],
                                 [0, 0, 4, 0.5],
                                 [0, 0, 5, 0.5]]))


def test_read_split_soma():
    n = load_neuron(SWC_PATH / 'split_soma_two_neurites.swc')

    assert_array_equal(n.soma.points,
                       [[1, 0, 1, 4.0],
                        [2, 0, 0, 4.0],
                        [3, 0, 0, 4.0]])

    assert len(n.neurites) == 2
    assert_array_equal(n.neurites[0].points,
                       [[0, 0, 2, 0.5],
                        [0, 0, 3, 0.5],
                        [0, 0, 4, 0.5],
                        [0, 0, 5, 0.5]])

    assert_array_equal(n.neurites[1].points,
                       [[0, 0, 6, 0.5],
                        [0, 0, 7, 0.5],
                        [0, 0, 8, 0.5],
                        [0, 0, 9, 0.5]])

    assert len(n.sections) == 2


def test_weird_indent():

    n = load_neuron("""

                 # this is the same as simple.swc

# but with a questionable styling

     1 1  0  0 0 1. -1
 2 3  0  0 0 1.  1

 3 3  0  5 0 1.  2
 4 3 -5  5 0 0.  3



 5 3  6  5 0 0.  3
     6 2  0  0 0 1.  1
 7 2  0 -4 0 1.  6

 8 2  6 -4 0         0.  7
 9 2 -5 -4 0 0.  7
""", reader='swc')

    simple = load_neuron(SWC_PATH / 'simple.swc')
    assert_array_equal(simple.points,
                       n.points)


def test_cyclic():
    with pytest.raises(RawDataError):
        load_neuron("""
        1 1  0  0 0 1. -1
        2 3  0  0 0 1.  1
        3 3  0  5 0 1.  2
        4 3 -5  5 0 0.  3
        5 3  6  5 0 0.  3
        6 2  0  0 0 1.  6  # <-- cyclic point
        7 2  0 -4 0 1.  6
        8 2  6 -4 0 0.  7
        9 2 -5 -4 0 0.  7""", reader='swc')


def test_simple_reversed():
    n = load_neuron(SWC_PATH / 'simple_reversed.swc')
    assert_array_equal(n.soma.points,
                       [[0, 0, 0, 1]])
    assert len(n.neurites) == 2
    assert len(n.neurites[0].points) == 4
    assert_array_equal(n.neurites[0].points,
                       [[0, 0, 0, 1],
                        [0, 5, 0, 1],
                        [-5, 5, 0, 0],
                        [6, 5, 0, 0]])
    assert_array_equal(n.neurites[1].points,
                       [[0, 0, 0, 1],
                        [0, -4, 0, 1],
                        [6, -4, 0, 0],
                        [-5, -4, 0, 0]])


def test_custom_type():
    neuron = load_neuron(Path(SWC_PATH, 'custom_type.swc'))
    assert neuron.neurites[1].type == NeuriteType.custom


def test_undefined_type():
    with pytest.raises(RawDataError, match='Unsupported section type: 0'):
        load_neuron(SWC_PATH / 'undefined_type.swc')
