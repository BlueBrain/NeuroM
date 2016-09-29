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

'''NeuroM neuron checking functions.

Contains functions for checking validity of neuron neurites and somata.
Tests assumes neurites and/or soma have been succesfully built where applicable,
i.e. soma- and neurite-related structural tests pass.
'''
import numpy as np

from itertools import izip
from neurom.core import Tree
from neurom.core.types import NeuriteType
from neurom.core.dataformat import COLS
from neurom.morphmath import section_length, segment_length
from neurom.check.morphtree import get_flat_neurites, get_nonmonotonic_neurites
from neurom.fst import _neuritefunc as _nf
from neurom.check import CheckResult


def _read_neurite_type(neurite):
    '''Simply read the stored neurite type'''
    return neurite.type


def has_axon(neuron, treefun=_read_neurite_type):
    '''Check if a neuron has an axon

    Arguments:
        neuron: The neuron object to test
        treefun: Optional function to calculate the tree type of
        neuron's neurites
    '''
    return CheckResult(NeuriteType.axon in [treefun(n) for n in neuron.neurites])


def has_apical_dendrite(neuron, min_number=1, treefun=_read_neurite_type):
    '''Check if a neuron has apical dendrites

    Arguments:
        neuron: The neuron object to test
        min_number: minimum number of apical dendrites required
        treefun: Optional function to calculate the tree type of neuron's
        neurites
    '''
    types = [treefun(n) for n in neuron.neurites]
    return CheckResult(types.count(NeuriteType.apical_dendrite) >= min_number)


def has_basal_dendrite(neuron, min_number=1, treefun=_read_neurite_type):
    '''Check if a neuron has basal dendrites

    Arguments:
        neuron: The neuron object to test
        min_number: minimum number of basal dendrites required
        treefun: Optional function to calculate the tree type of neuron's
        neurites
    '''
    types = [treefun(n) for n in neuron.neurites]
    return CheckResult(types.count(NeuriteType.basal_dendrite) >= min_number)


def has_no_flat_neurites(neuron, tol=0.1, method='ratio'):
    '''Check that a neuron has no flat neurites

    Argument:
        neuron : The neuron object to test
        tol : the tolerance or the ratio
        method : way of determining flatness, 'tolerance', 'ratio'
    '''
    return CheckResult(len(get_flat_neurites(neuron, tol, method)) == 0)


def has_all_monotonic_neurites(neuron, tol=1e-6):
    '''Check that a neuron has no neurites that are not monotonic

    Argument:
        neuron : The neuron object to test
        tol : the tolerance for testing monotonicity
    '''
    return CheckResult(len(get_nonmonotonic_neurites(neuron, tol)) == 0)


def has_all_nonzero_segment_lengths(neuron, threshold=0.0):
    '''Check presence of neuron segments with length not above threshold

    Arguments:
        neuron: Neuron object whose segments will be tested
        threshold: value above which a segment length is considered to be non-zero

    Return:
        status and list of (section_id, segment_id) of zero length segments
    '''
    bad_ids = []
    for sec in _nf.iter_sections(neuron):
        p = sec.points
        for i, s in enumerate(izip(p[:-1], p[1:])):
            if segment_length(s) <= threshold:
                bad_ids.append((sec.id, i))

    return CheckResult(len(bad_ids) == 0, bad_ids)


def has_all_nonzero_section_lengths(neuron, threshold=0.0):
    '''Check presence of neuron sections with length not above threshold

    Arguments:
        neuron: Neuron object whose segments will be tested
        threshold: value above which a section length is considered to be non-zero

    Return:
        status and list of ids bad sections
    '''
    bad_ids = [s.id for s in _nf.iter_sections(neuron.neurites)
               if section_length(s.points) <= threshold]

    return CheckResult(len(bad_ids) == 0, bad_ids)


def has_all_nonzero_neurite_radii(neuron, threshold=0.0):
    '''Check presence of neurite points with radius not above threshold

    Arguments:
        neuron: Neuron object whose segments will be tested
        threshold: value above which a radius is considered to be non-zero
    Return:
        status and list of (section ID, point ID) pairs of zero-radius points
    '''
    bad_ids = []
    seen_ids = set()
    for s in _nf.iter_sections(neuron):
        for i, p in enumerate(s.points):
            info = (s.id, i)
            if p[COLS.R] <= threshold and info not in seen_ids:
                seen_ids.add(info)
                bad_ids.append(info)

    return CheckResult(len(bad_ids) == 0, bad_ids)


def has_nonzero_soma_radius(neuron, threshold=0.0):
    '''Check if soma radius not above threshold

    Arguments:
        neuron: Neuron object whose soma will be tested
        threshold: value above which the soma radius is considered to be non-zero
    '''
    return CheckResult(neuron.soma.radius > threshold)


def has_no_jumps(neuron, max_distance=30.0, axis='z'):
    '''Check if there are jumps (large movements in the `axis`)

    Arguments:
        neuron: Neuron object whose neurites will be tested
        max_z_distance: value above which consecutive z-values are
        considered a jump
        axis(str): one of x/y/z, which axis to check for jumps


    Return:
        status and list of ids bad sections
    '''
    bad_ids = []
    axis = {'x': COLS.X, 'y': COLS.Y, 'z': COLS.Z, }[axis.lower()]
    for s in _nf.iter_sections(neuron):
        it = izip(s.points, s.points[1:])
        for i, (p0, p1) in enumerate(it):
            info = (s.id, i)
            if max_distance < abs(p0[axis] - p1[axis]):
                bad_ids.append(info)
    return CheckResult(len(bad_ids) == 0, bad_ids)


def has_no_fat_ends(neuron, multiple_of_mean=2.0, final_point_count=5):
    '''Check if leaf points are too large

    Arguments:
        neuron: Neuron object whose soma will be tested
        multiple_of_mean(float): how many times larger the final radius
        has to be compared to the mean of the final points
        final_point_count(int): how many points to include in the mean
    '''
    bad_ids = []
    for leaf in _nf.iter_sections(neuron.neurites, iterator_type=Tree.ileaf):
        mean_radius = np.mean(leaf.points[-final_point_count:, COLS.R])
        if mean_radius * multiple_of_mean < leaf.points[-1, COLS.R]:
            bad_ids.append((leaf.id, len(leaf.points)))

    return CheckResult(len(bad_ids) == 0, bad_ids)
