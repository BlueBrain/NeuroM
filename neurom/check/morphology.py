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

'''
Python module of NeuroM to check neurons.
'''

from neurom.core.types import TreeType
from neurom.core.tree import ipreorder
from neurom.core.tree import isegment
from neurom.core.tree import isection
from neurom.core.tree import val_iter
from neurom.core.dataformat import COLS
from neurom.analysis.morphmath import section_length
from neurom.analysis.morphmath import segment_length
from neurom.analysis.morphtree import find_tree_type
from itertools import chain


def has_axon(neuron, treefun=find_tree_type):
    '''Check if a neuron has an axon

    Arguments:
        neuron: The neuron object to test
        treefun: Optional function to calculate the tree type of
        neuron's neurites
    '''
    return TreeType.axon in [treefun(n) for n in neuron.neurites]


def has_apical_dendrite(neuron, min_number=1, treefun=find_tree_type):
    '''Check if a neuron has apical dendrites

    Arguments:
        neuron: The neuron object to test
        min_number: minimum number of apical dendrites required
        treefun: Optional function to calculate the tree type of neuron's
        neurites
    '''
    types = [treefun(n) for n in neuron.neurites]
    return types.count(TreeType.apical_dendrite) >= min_number


def has_basal_dendrite(neuron, min_number=1, treefun=find_tree_type):
    '''Check if a neuron has basal dendrites

    Arguments:
        neuron: The neuron object to test
        min_number: minimum number of basal dendrites required
        treefun: Optional function to calculate the tree type of neuron's
        neurites
    '''
    types = [treefun(n) for n in neuron.neurites]
    return types.count(TreeType.basal_dendrite) >= min_number


def nonzero_segment_lengths(neuron, threshold=0.0):
    '''Check presence of neuron segments with length not above threshold

    Arguments:
        neuron: Neuron object whose segments will be tested
        threshold: value above which a segment length is considered to be non-zero
    Return: list of (first_id, second_id) of zero length segments
    '''
    l = [[s for s in val_iter(isegment(t))
          if segment_length(s) <= threshold]
         for t in neuron.neurites]
    return [(i[0][COLS.ID], i[1][COLS.ID]) for i in chain(*l)]


def nonzero_section_lengths(neuron, threshold=0.0):
    '''Check presence of neuron sections with length not above threshold

    Arguments:
        neuron: Neuron object whose segments will be tested
        threshold: value above which a section length is considered to be non-zero
    Return: list of ids of first point in bad sections
    '''
    l = [[s for s in val_iter(isection(t))
          if section_length(s) <= threshold]
         for t in neuron.neurites]
    return [i[0][COLS.ID] for i in chain(*l)]


def nonzero_neurite_radii(neuron, threshold=0.0):
    '''Check presence of neurite points with radius not above threshold

    Arguments:
        neuron: Neuron object whose segments will be tested
        threshold: value above which a radius is considered to be non-zero
    Return: list of IDs of zero-radius points
    '''

    ids = [[i[COLS.ID] for i in val_iter(ipreorder(t))
            if i[COLS.R] <= threshold] for t in neuron.neurites]
    return [i for i in chain(*ids)]
