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

from neurom.core.types import NeuriteType
from neurom.core.tree import ipreorder
from neurom.core.tree import isegment
from neurom.core.tree import isection
from neurom.core.tree import val_iter
from neurom.core.dataformat import COLS
from neurom.analysis.morphmath import section_length
from neurom.analysis.morphmath import segment_length
from neurom.analysis.morphtree import find_tree_type
from neurom.check.morphtree import is_flat, is_monotonic, is_back_tracking
from itertools import chain


def has_axon(neuron, treefun=find_tree_type):
    '''Check if a neuron has an axon

    Arguments:
        neuron: The neuron object to test
        treefun: Optional function to calculate the tree type of
        neuron's neurites
    '''
    return NeuriteType.axon in [treefun(n) for n in neuron.neurites]


def has_apical_dendrite(neuron, min_number=1, treefun=find_tree_type):
    '''Check if a neuron has apical dendrites

    Arguments:
        neuron: The neuron object to test
        min_number: minimum number of apical dendrites required
        treefun: Optional function to calculate the tree type of neuron's
        neurites
    '''
    types = [treefun(n) for n in neuron.neurites]
    return types.count(NeuriteType.apical_dendrite) >= min_number


def has_basal_dendrite(neuron, min_number=1, treefun=find_tree_type):
    '''Check if a neuron has basal dendrites

    Arguments:
        neuron: The neuron object to test
        min_number: minimum number of basal dendrites required
        treefun: Optional function to calculate the tree type of neuron's
        neurites
    '''
    types = [treefun(n) for n in neuron.neurites]
    return types.count(NeuriteType.basal_dendrite) >= min_number


def get_flat_neurites(neuron, tol=0.1, method='ratio'):
    '''Check if a neuron has neurites that are flat within a tolerance

    Argument:
        neuron : The neuron object to test
        tol : the tolerance or the ratio
        method : way of determining flatness, 'tolerance', 'ratio'

    Returns:
        Bool list corresponding to the flatness check for each neurite
        in neuron neurites with respect to the given criteria
    '''
    return [n for n in neuron.neurites if is_flat(n, tol, method)]


def has_no_flat_neurites(neuron, tol=0.1, method='ratio'):
    '''Check that a neuron has no flat neurites

    Argument:
        neuron : The neuron object to test
        tol : the tolerance or the ratio
        method : way of determining flatness, 'tolerance', 'ratio'
    '''
    return len(get_flat_neurites(neuron, tol, method)) == 0


def get_nonmonotonic_neurites(neuron, tol=1e-6):
    '''Get neurites that are not monotonic

    Argument:
        neuron : The neuron object to test
        tol : the tolerance for testing monotonicity

    Returns:
        list of neurites that do not satisfy monotonicity test
    '''
    return [n for n in neuron.neurites if not is_monotonic(n, tol)]


def has_all_monotonic_neurites(neuron, tol=1e-6):
    '''Check that a neuron has no neurites that are not monotonic

    Argument:
        neuron : The neuron object to test
        tol : the tolerance for testing monotonicity
    '''
    return len(get_nonmonotonic_neurites(neuron, tol)) == 0


def get_back_tracking_neurites(neuron):
    '''Get neurites that have back-tracks. A back-track is the placement of
    a point near a previous segment during the reconstruction, causing
    a zigzag jump in the morphology which can cause issues with meshing
    algorithms.
    '''
    return [n for n in neuron.neurites if is_back_tracking(n)]


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
