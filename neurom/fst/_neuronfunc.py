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

'''Morphometrics functions for neurons or neuron populations'''

import math
import numpy as np
from neurom.geom import bounding_box
from neurom.core.types import NeuriteType
from neurom.core.types import tree_type_checker as is_type
from neurom.core.dataformat import COLS
from neurom.core._neuron import iter_neurites, iter_segments
from neurom import morphmath


def neuron_population(nrns):
    '''Makes sure `nrns` behaves like a neuron population'''
    return nrns.neurons if hasattr(nrns, 'neurons') else (nrns,)


def soma_surface_area(nrn, neurite_type=NeuriteType.soma):
    '''Get the surface area of a neuron's soma.

    Note:
        The surface area is calculated by assuming the soma is spherical.
    '''
    assert neurite_type == NeuriteType.soma, 'Neurite type must be soma'
    return 4 * math.pi * nrn.soma.radius ** 2


def soma_surface_areas(nrn_pop, neurite_type=NeuriteType.soma):
    '''Get the surface areas of the somata in a population of neurons

    Note:
        The surface area is calculated by assuming the soma is spherical.
    Note:
        If a single neuron is passed, a single element list with the surface
        area of its soma member is returned.
    '''
    nrns = neuron_population(nrn_pop)
    assert neurite_type == NeuriteType.soma, 'Neurite type must be soma'
    return [soma_surface_area(n) for n in nrns]


def soma_radii(nrn_pop, neurite_type=NeuriteType.soma):
    ''' Get the radii of the somata of a population of neurons

    Note:
        If a single neuron is passed, a single element list with the
        radius of its soma member is returned.
    '''
    assert neurite_type == NeuriteType.soma, 'Neurite type must be soma'
    nrns = neuron_population(nrn_pop)
    return [n.soma.radius for n in nrns]


def trunk_section_lengths(nrn, neurite_type=None):
    '''list of lengths of trunk sections of neurites in a neuron'''
    neurite_filter = is_type(neurite_type)
    return [morphmath.section_length(s.root_node.points)
            for s in nrn.neurites if neurite_filter(s)]


def trunk_origin_radii(nrn, neurite_type=None):
    '''radii of the trunk sections of neurites in a neuron'''
    neurite_filter = is_type(neurite_type)
    return [s.root_node.points[0][COLS.R] for s in nrn.neurites if neurite_filter(s)]


def trunk_origin_azimuths(nrn, neurite_type=None):
    '''Get a list of all the trunk origin azimuths of a neuron or population

    The azimuth is defined as Angle between x-axis and the vector
    defined by (initial tree point - soma center) on the x-z plane.

    The range of the azimuth angle [-pi, pi] radians
    '''
    neurite_filter = is_type(neurite_type)
    nrns = neuron_population(nrn)

    def _azimuth(section, soma):
        '''Azimuth of a section'''
        vector = morphmath.vector(section[0], soma.center)
        return np.arctan2(vector[COLS.Z], vector[COLS.X])

    return [_azimuth(s.root_node.points, n.soma)
            for n in nrns
            for s in n.neurites if neurite_filter(s)]


def trunk_origin_elevations(nrn, neurite_type=None):
    '''Get a list of all the trunk origin elevations of a neuron or population

    The elevation is defined as the angle between x-axis and the
    vector defined by (initial tree point - soma center)
    on the x-y half-plane.

    The range of the elevation angle [-pi/2, pi/2] radians
    '''
    neurite_filter = is_type(neurite_type)
    nrns = neuron_population(nrn)

    def _elevation(section, soma):
        '''Elevation of a section'''
        vector = morphmath.vector(section[0], soma.center)
        norm_vector = np.linalg.norm(vector)

        if norm_vector >= np.finfo(type(norm_vector)).eps:
            return np.arcsin(vector[COLS.Y] / norm_vector)
        else:
            raise ValueError("Norm of vector between soma center and section is almost zero.")

    return [_elevation(s.root_node.points, n.soma)
            for n in nrns
            for s in n.neurites if neurite_filter(s)]


def sholl_crossings(neurites, center, radii):
    '''calculate crossings of neurites

    Args:
        nrn(morph): morphology on which to perform Sholl analysis
        radii(iterable of floats): radii for which crossings will be counted

    Returns:
        Array of same length as radii, with a count of the number of crossings
        for the respective radius
    '''
    def _count_crossings(neurite, radius):
        '''count_crossings of segments in neurite with radius'''
        r2 = radius ** 2
        count = 0
        for start, end in iter_segments(neurite):
            start_dist2, end_dist2 = (morphmath.point_dist2(center, start),
                                      morphmath.point_dist2(center, end))

            count += int(start_dist2 <= r2 <= end_dist2 or
                         end_dist2 <= r2 <= start_dist2)

        return count

    return np.array([sum(_count_crossings(neurite, r)
                         for neurite in iter_neurites(neurites))
                     for r in radii])


def sholl_frequency(nrn, neurite_type=None, step_size=10):
    '''perform Sholl frequency calculations on a population of neurites

    Args:
        nrn(morph): nrn or population
        neurite_type(NeuriteType): which neurites to operate on
        step_size(float): step size between Sholl radii

    Note:
        Given a neuron, the soma center is used for the concentric circles,
        which range from the soma radii, and the maximum radial distance
        in steps of `step_size`.  When a population is given, the concentric
        circles range from the smallest soma radius to the largest radial neurite
        distance.  Finally, each segment of the neuron is tested, so a neurite that
        bends back on itself, and crosses the same Sholl radius will get counted as
        having crossed multiple times.
    '''
    nrns = neuron_population(nrn)
    neurite_filter = is_type(neurite_type)

    min_soma_edge = float('Inf')
    max_radii = 0
    neurites_list = []
    for neuron in nrns:
        neurites_list.extend(((neurites, neuron.soma.center)
                              for neurites in neuron.neurites
                              if neurite_filter(neurites)))

        min_soma_edge = min(min_soma_edge, neuron.soma.radius)
        max_radii = max(max_radii, np.max(np.abs(bounding_box(neuron))))

    radii = np.arange(min_soma_edge, max_radii + step_size, step_size)
    ret = np.zeros_like(radii)
    for neurites, center in neurites_list:
        ret += sholl_crossings(neurites, center, radii)

    return ret
