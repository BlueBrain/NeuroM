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

import os

import numpy as np

from neurom import load_neuron
from neurom.core.dataformat import COLS
from numpy.testing import assert_array_equal

from nose import tools as nt
from neurom.exceptions import SomaError, RawDataError


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')
SWC_SOMA_PATH = os.path.join(SWC_PATH, 'soma')


@nt.raises(RawDataError)
def test_repeated_id():
    n = load_neuron(os.path.join(SWC_PATH, 'repeated_id.swc'))

@nt.raises(SomaError)
def test_neurite_followed_by_soma():
    load_neuron(os.path.join(SWC_PATH, 'soma_with_neurite_parent.swc'))

def test_read_single_neurite():
    n = load_neuron(os.path.join(SWC_PATH, 'point_soma_single_neurite.swc'))
    nt.eq_(len(n.neurites), 1)
    nt.eq_(n.neurites[0].root_node.id, 0)
    assert_array_equal(n.soma.points,
                       [[0,0,0,3.0]])
    nt.eq_(len(n.neurites), 1)
    nt.eq_(len(n.sections), 1)
    assert_array_equal(n.neurites[0].points,
                       np.array([[0,0,2,0.5],
                                 [0,0,3,0.5],
                                 [0,0,4,0.5],
                                 [0,0,5,0.5]]))


def test_read_split_soma():
    n = load_neuron(os.path.join(SWC_PATH, 'split_soma_two_neurites.swc'))

    assert_array_equal(n.soma.points,
                       [[1,0,1,4.0],
                        [2,0,0,4.0],
                        [3,0,0,4.0]])

    nt.assert_equal(len(n.neurites), 2)
    assert_array_equal(n.neurites[0].points,
                       [[0,0,2,0.5],
                        [0,0,3,0.5],
                        [0,0,4,0.5],
                        [0,0,5,0.5]])

    assert_array_equal(n.neurites[1].points,
                       [[0,0,6,0.5],
                        [0,0,7,0.5],
                        [0,0,8,0.5],
                        [0,0,9,0.5]])

    nt.eq_(len(n.sections), 2)


def test_simple_reversed():
    n = load_neuron(os.path.join(SWC_PATH, 'whitespaces.swc'))


def test_simple_reversed():
    n = load_neuron(os.path.join(SWC_PATH, 'simple_reversed.swc'))
    assert_array_equal(n.soma.points,
                       [[0,0,0,1]])
    nt.assert_equal(len(n.neurites), 2)
    nt.assert_equal(len(n.neurites[0].points), 4)
    assert_array_equal(n.neurites[0].points,
                       [[0,0,0,1],
                        [0,5,0,1],
                        [-5,5,0,0],
                        [6,5,0,0]])
    assert_array_equal(n.neurites[1].points,
                       [[0,0,0,1],
                        [0,-4,0,1],
                        [6,-4,0,0],
                        [-5,-4,0,0]])
