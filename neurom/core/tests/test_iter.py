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
from io import StringIO

from nose import tools as nt
from numpy.testing import assert_array_equal

import neurom as nm
from neurom import COLS, core, load_neuron
from neurom.core import NeuriteIter, Tree

DATA_PATH = Path(__file__).parent.parent.parent.parent / 'test_data'

NRN1 = load_neuron(Path(DATA_PATH, 'swc/Neuron.swc'))

NEURONS = [NRN1,
           load_neuron(Path(DATA_PATH, 'swc/Single_basal.swc')),
           load_neuron(Path(DATA_PATH, 'swc/Neuron_small_radius.swc')),
           load_neuron(Path(DATA_PATH, 'swc/Neuron_3_random_walker_branches.swc')),
           ]
TOT_NEURITES = sum(len(N.neurites) for N in NEURONS)

REVERSED_NEURITES = load_neuron(Path(DATA_PATH, 'swc/ordering/reversed_NRN_neurite_order.swc'))

POP = core.Population(NEURONS, name='foo')


def assert_sequence_equal(a, b):
    nt.eq_(tuple(a), tuple(b))


def test_iter_neurites_default():
    assert_sequence_equal(POP.neurites,
                          [n for n in core.iter_neurites(POP)])


def test_iter_neurites_nrn_order():
    assert_sequence_equal(list(core.iter_neurites(REVERSED_NEURITES,
                                                  neurite_order=NeuriteIter.NRN)),
                          reversed(list(core.iter_neurites(REVERSED_NEURITES))))


def test_iter_neurites_filter():

    for ntyp in nm.NEURITE_TYPES:
        a = [n for n in POP.neurites if n.type == ntyp]
        b = [n for n in core.iter_neurites(POP, filt=lambda n: n.type == ntyp)]
        assert_sequence_equal(a, b)


def test_iter_neurites_mapping():

    n = [n for n in core.iter_neurites(POP, mapfun=lambda n: len(n.points))]
    ref = [211, 211, 211, 211, 211, 211, 211, 211, 211, 500, 500, 500]
    assert_sequence_equal(n, ref)


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
        a = [s.id for n in filter(lambda nn: nn.type == ntyp, POP.neurites)
             for s in n.iter_sections()]
        b = [n.id for n in core.iter_sections(POP, neurite_filter=lambda n: n.type == ntyp)]
        assert_sequence_equal(a, b)

def test_iter_sections_inrnorder():
    assert_sequence_equal([s.id for n in POP.neurites for s in n.iter_sections(neurite_order=NeuriteIter.NRN)],
                          [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 2, 3, 4])

def test_iter_sections_ipreorder():
    assert_sequence_equal([s.id for n in POP.neurites for s in n.iter_sections(Tree.ipreorder)],
                          [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 2, 3, 4])


def test_iter_sections_ipostorder():
    assert_sequence_equal([s.id for n in POP.neurites for s in n.iter_sections(Tree.ipostorder)],
                          [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 85, 83, 81, 79, 77, 75, 73, 71, 69, 67, 65, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 85, 83, 81, 79, 77, 75, 73, 71, 69, 67, 65, 2, 3, 4])


def test_iter_sections_ibifurcation():
    assert_sequence_equal([s.id for n in POP.neurites for s in n.iter_sections(Tree.ibifurcation_point)],
                          [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83],)


def test_iter_sections_iforking():
    assert_sequence_equal([s.id for n in POP.neurites for s in n.iter_sections(Tree.iforking_point)],
                          [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83])


def test_iter_sections_ileaf():
    assert_sequence_equal([s.id for n in POP.neurites for s in n.iter_sections(Tree.ileaf)],
                          [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 85, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 85, 2, 3, 4])


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
    sec = load_neuron(StringIO(u"""
	                      ((CellBody)
	                       (0 0 0 2))

	                      ((Dendrite)
                          (1 2 3 8)
                          (5 6 7 16)
                          (8 7 6 10)
                          (4 3 2 2))
                       """), reader='asc').sections[1]
    ref = [[p1[COLS.XYZR].tolist(), p2[COLS.XYZR].tolist()]
           for p1, p2 in core.iter_segments(sec)]

    assert_array_equal(ref, [[[1, 2, 3, 4], [5, 6, 7, 8]],
                             [[5, 6, 7, 8], [8, 7, 6, 5]],
                             [[8, 7, 6, 5], [4, 3, 2, 1]]])
