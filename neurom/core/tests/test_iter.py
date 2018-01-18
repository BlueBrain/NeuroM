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
from os.path import join as joinp

from nose import tools as nt
import neurom as nm
from neurom import load_neuron
from neurom import core
from neurom.core import Tree
from neurom._compat import filter

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = joinp(_path, '../../../test_data')

NRN1 = load_neuron(joinp(DATA_PATH, 'swc/Neuron.swc'))
NEURONS = [NRN1,
           load_neuron(joinp(DATA_PATH, 'swc/Single_basal.swc')),
           load_neuron(joinp(DATA_PATH, 'swc/Neuron_small_radius.swc')),
           load_neuron(joinp(DATA_PATH, 'swc/Neuron_3_random_walker_branches.swc')),
           ]
TOT_NEURITES = sum(len(N.neurites) for N in NEURONS)
POP = core.Population(NEURONS, name='foo')


def assert_sequence_equal(a, b):
    nt.eq_(tuple(a), tuple(b))


def test_iter_neurites_default():
    assert_sequence_equal(POP.neurites,
                          [n for n in core.iter_neurites(POP)])


def test_iter_neurites_filter():

    for ntyp in nm.NEURITE_TYPES:
        a = [n for n in POP.neurites if n.type == ntyp]
        b = [n for n in core.iter_neurites(POP, filt=lambda n: n.type == ntyp)]
        assert_sequence_equal(a, b)


def test_iter_neurites_mapping():
    assert_sequence_equal(list(core.iter_neurites(POP, mapfun=lambda n: len(n.points))),
                          [211, 211, 211, 211, 211, 211, 211, 211, 211, 500, 500, 500])


def test_iter_neurites_filter_mapping():
    n = [n for n in core.iter_neurites(POP,
                                       mapfun=lambda n: len(n.points),
                                       filt=lambda n: len(n.points) > 250)]

    ref = [500, 500, 500]
    assert_sequence_equal(n, ref)


def test_iter_sections_default():

    ref = [s for n in POP.neurites for s in n.iter_sections()]
    assert_sequence_equal(ref,
                          [n for n in core.iter_sections(POP)])


def test_iter_sections_filter():

    for ntyp in nm.NEURITE_TYPES:
        a = [s for n in filter(lambda nn: nn.type == ntyp, POP.neurites)
             for s in n.iter_sections()]
        b = [n for n in core.iter_sections(POP, neurite_filter=lambda n : n.type == ntyp)]
        assert_sequence_equal(a, b)


def test_iter_sections_ipostorder():

    ref = [s for n in POP.neurites for s in n.iter_sections(Tree.ipostorder)]
    assert_sequence_equal(ref,
                          [n for n in core.iter_sections(POP, iterator_type=Tree.ipostorder)])


def test_iter_sections_ibifurcation():

    ref = [s for n in POP.neurites for s in n.iter_sections(Tree.ibifurcation_point)]
    assert_sequence_equal(ref,
                          [n for n in core.iter_sections(POP, iterator_type=Tree.ibifurcation_point)])


def test_iter_sections_iforking():

    ref = [s for n in POP.neurites for s in n.iter_sections(Tree.iforking_point)]
    assert_sequence_equal(ref,
                          [n for n in core.iter_sections(POP, iterator_type=Tree.iforking_point)])


def test_iter_sections_ileaf():

    ref = [s for n in POP.neurites for s in n.iter_sections(Tree.ileaf)]
    assert_sequence_equal(ref,
                          [n for n in core.iter_sections(POP, iterator_type=Tree.ileaf)])


def test_iter_segments_nrn():

    ref = list(core.iter_segments(NRN1))
    nt.eq_(len(ref), 840)

    ref = list(core.iter_segments(NRN1, neurite_filter=lambda n: n.type == nm.AXON))
    nt.eq_(len(ref), 210)

    ref = list(core.iter_segments(NRN1, neurite_filter=lambda n: n.type == nm.BASAL_DENDRITE))
    nt.eq_(len(ref), 420)

    ref = list(core.iter_segments(NRN1, neurite_filter=lambda n: n.type == nm.APICAL_DENDRITE))
    nt.eq_(len(ref), 210)


def test_iter_segments_pop():

    ref = list(core.iter_segments(POP))
    nt.eq_(len(ref), 3387)

    ref = list(core.iter_segments(POP, neurite_filter=lambda n: n.type == nm.AXON))
    nt.eq_(len(ref), 919)

    ref = list(core.iter_segments(POP, neurite_filter=lambda n: n.type == nm.BASAL_DENDRITE))
    nt.eq_(len(ref), 1549)

    ref = list(core.iter_segments(POP, neurite_filter=lambda n: n.type == nm.APICAL_DENDRITE))
    nt.eq_(len(ref), 919)


def test_iter_segments_section():
    sec = core.Section([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [8, 7, 6, 5],
                        [4, 3, 2, 1],
                        ])
    ref = list(core.iter_segments(sec))
    nt.eq_(ref, [([1, 2, 3, 4], [5, 6, 7, 8]),
                 ([5, 6, 7, 8], [8, 7, 6, 5]),
                 ([8, 7, 6, 5], [4, 3, 2, 1])])
