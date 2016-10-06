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
from itertools import ifilter

from nose import tools as nt
import neurom as nm
from neurom.core.population import Population
from neurom import load_neuron
from neurom import core
from neurom.core import Tree

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = joinp(_path, '../../../test_data')

NRN1 = load_neuron(joinp(DATA_PATH, 'swc/Neuron.swc'))
NRN2 = load_neuron(joinp(DATA_PATH, 'swc/Single_basal.swc'))
NRN3 = load_neuron(joinp(DATA_PATH, 'swc/Neuron_small_radius.swc'))
NRN4 = load_neuron(joinp(DATA_PATH, 'swc/Neuron_3_random_walker_branches.swc'))

NEURONS = [NRN1, NRN2, NRN3, NRN4]
TOT_NEURITES = sum(len(N.neurites) for N in NEURONS)
POP = Population(NEURONS, name='foo')

def test_iter_neurites_default():

    nt.assert_sequence_equal(POP.neurites,
                             [n for n in core.iter_neurites(POP)])

def test_iter_neurites_filter():

    for ntyp in nm.NEURITE_TYPES:
        a = [n for n in POP.neurites if n.type == ntyp]
        b = [n for n in core.iter_neurites(POP, filt=lambda n : n.type == ntyp)]
        nt.assert_sequence_equal(a, b)


def test_iter_neurites_mapping():

    n = [n for n in core.iter_neurites(POP, mapfun=lambda n : len(n.points))]
    ref = [211, 211, 211, 211, 211, 211, 211, 211, 211, 500, 500, 500]
    nt.assert_sequence_equal(n, ref)


def test_iter_neurites_filter_mapping():
    n = [n for n in core.iter_neurites(POP,
                                       mapfun=lambda n : len(n.points),
                                       filt=lambda n : len(n.points) > 250)]

    ref = [500, 500, 500]
    nt.assert_sequence_equal(n, ref)


def test_iter_sections_default():

    ref = [s for n in POP.neurites for s in n.iter_sections()]
    nt.assert_sequence_equal(ref,
                             [n for n in core.iter_sections(POP)])


def test_iter_sections_filter():

    for ntyp in nm.NEURITE_TYPES:
        a = [s for n in ifilter(lambda nn: nn.type == ntyp, POP.neurites)
             for s in n.iter_sections()]
        b = [n for n in core.iter_sections(POP, neurite_filter=lambda n : n.type == ntyp)]
        nt.assert_sequence_equal(a, b)


def test_iter_sections_ipostorder():

    ref = [s for n in POP.neurites for s in n.iter_sections(Tree.ipostorder)]
    nt.assert_sequence_equal(ref,
                             [n for n in core.iter_sections(POP, iterator_type=Tree.ipostorder)])


def test_iter_sections_ibifurcation():

    ref = [s for n in POP.neurites for s in n.iter_sections(Tree.ibifurcation_point)]
    nt.assert_sequence_equal(ref,
                             [n for n in core.iter_sections(POP, iterator_type=Tree.ibifurcation_point)])


def test_iter_sections_iforking():

    ref = [s for n in POP.neurites for s in n.iter_sections(Tree.iforking_point)]
    nt.assert_sequence_equal(ref,
                             [n for n in core.iter_sections(POP, iterator_type=Tree.iforking_point)])


def test_iter_sections_ileaf():

    ref = [s for n in POP.neurites for s in n.iter_sections(Tree.ileaf)]
    nt.assert_sequence_equal(ref,
                             [n for n in core.iter_sections(POP, iterator_type=Tree.ileaf)])
