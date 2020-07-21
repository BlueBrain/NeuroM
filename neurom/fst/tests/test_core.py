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

"""Test neurom.fst._core module."""

from pathlib import Path
from copy import deepcopy

import numpy as np
from nose import tools as nt

from neurom import io as _io
from neurom.fst import _core

DATA_ROOT = Path(__file__).parent.parent.parent.parent / 'test_data'
DATA_PATH = Path(DATA_ROOT, 'valid_set')
FILENAMES = [Path(DATA_PATH, f)
             for f in ['Neuron.swc', 'Neuron_h5v1.h5', 'Neuron_h5v2.h5']]


def test_neuron_name():

    d = _io.load_data(FILENAMES[0])
    nrn = _core.FstNeuron(d, '12af3rg')
    nt.eq_(nrn.name, '12af3rg')


def test_section_str():
    s = _core.Section('foo')
    nt.assert_true(isinstance(str(s), str))


def _check_cloned_neurites(a, b):

    nt.assert_true(a is not b)
    nt.assert_true(a.root_node is not b.root_node)
    nt.assert_equal(a.type, b.type)
    for aa, bb in zip(a.iter_sections(), b.iter_sections()):
        nt.assert_true(np.all(aa.points == bb.points))


def test_neuron_deepcopy():

    d = _io.load_neuron(FILENAMES[0])
    dc = deepcopy(d)

    nt.assert_true(d is not dc)

    nt.assert_true(d.soma is not dc.soma)

    nt.assert_true(np.all(d.soma.points == dc.soma.points))
    nt.assert_true(np.all(d.soma.center == dc.soma.center))
    nt.assert_equal(d.soma.radius, dc.soma.radius)

    for a, b in zip(d.neurites, dc.neurites):
        _check_cloned_neurites(a, b)


def test_neurite_deepcopy():

    d = _io.load_neuron(FILENAMES[0])
    nrt = d.neurites[0]
    nrt2 = deepcopy(nrt)

    nt.assert_true(nrt is not nrt2)

    _check_cloned_neurites(nrt, nrt2)
