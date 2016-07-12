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

from itertools import izip
from neurom.core.types import NeuriteType
from neurom.core.neuron import make_soma
from neurom.core.dataformat import COLS
from neurom.analysis.morphmath import section_length
from neurom.analysis.morphmath import segment_length
from neurom.check.morphtree import is_flat, is_monotonic, is_back_tracking
from neurom.exceptions import SomaError
from neurom.fst import _neuritefunc as _nf


def _read_neurite_type(neurite):
    '''Simply read the stored neurite type'''
    return neurite.type


def has_valid_soma(data_wrapper):
    '''Check if a data block has a valid soma'''
    try:
        make_soma(data_wrapper.soma_points())
        return True
    except SomaError:
        return False


def has_axon(neuron, treefun=_read_neurite_type):
    '''Check if a neuron has an axon

    Arguments:
        neuron: The neuron object to test
        treefun: Optional function to calculate the tree type of
        neuron's neurites
    '''
    return NeuriteType.axon in [treefun(n) for n in neuron.neurites]


def has_apical_dendrite(neuron, min_number=1, treefun=_read_neurite_type):
    '''Check if a neuron has apical dendrites

    Arguments:
        neuron: The neuron object to test
        min_number: minimum number of apical dendrites required
        treefun: Optional function to calculate the tree type of neuron's
        neurites
    '''
    types = [treefun(n) for n in neuron.neurites]
    return types.count(NeuriteType.apical_dendrite) >= min_number


def has_basal_dendrite(neuron, min_number=1, treefun=_read_neurite_type):
    '''Check if a neuron has basal dendrites

    Arguments:
        neuron: The neuron object to test
        min_number: minimum number of basal dendrites required
        treefun: Optional function to calculate the tree type of neuron's
        neurites
    '''
    types = [treefun(n) for n in neuron.neurites]
    return types.count(NeuriteType.basal_dendrite) >= min_number


def has_nonzero_soma_radius(neuron, threshold=0.0):
    '''Check if soma radius not above threshold

    Arguments:
        neuron: Neuron object whose soma will be tested
        threshold: value above which the soma radius is considered to be non-zero
    '''
    return neuron.soma.radius > threshold


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
    Return: list of (section_id, segment_id) of zero length segments
    '''
    bad_ids = []
    for sec in _nf.iter_sections(neuron):
        p = sec.points
        for i, s in enumerate(izip(p[:-1], p[1:])):
            if segment_length(s) <= threshold:
                bad_ids.append((sec.id, i))

    return bad_ids


def nonzero_section_lengths(neuron, threshold=0.0):
    '''Check presence of neuron sections with length not above threshold

    Arguments:
        neuron: Neuron object whose segments will be tested
        threshold: value above which a section length is considered to be non-zero
    Return: list of ids bad sections
    '''
    return [s.id for s in _nf.iter_sections(neuron.neurites)
            if section_length(s.points) <= threshold]


def nonzero_neurite_radii(neuron, threshold=0.0):
    '''Check presence of neurite points with radius not above threshold

    Arguments:
        neuron: Neuron object whose segments will be tested
        threshold: value above which a radius is considered to be non-zero
    Return: list of (section ID, point ID) pairs of zero-radius points
    '''
    bad_ids = []
    seen_ids = set()
    for s in _nf.iter_sections(neuron):
        for i, p in enumerate(s.points):
            info = (s.id, i)
            if p[COLS.R] <= threshold and info not in seen_ids:
                seen_ids.add(info)
                bad_ids.append(info)

    return bad_ids
