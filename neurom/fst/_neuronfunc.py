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
from neurom.core.types import NeuriteType
from neurom.core.types import tree_type_checker as is_type
from neurom.core.dataformat import COLS
from neurom import morphmath as mm


def soma_surface_area(nrn):
    '''Get the surface area of a neuron's soma.

    Note:
        The surface area is calculated by assuming the soma is spherical.
    '''
    return 4 * math.pi * nrn.soma.radius ** 2


def soma_surface_areas(nrn_pop):
    '''Get the surface areas of the somata in a population of neurons

    Note:
        The surface area is calculated by assuming the soma is spherical.
    Note:
        If a single neuron is passed, a single element list with the surface
        area of its soma member is returned.
    '''
    nrns = nrn_pop.neurons if hasattr(nrn_pop, 'neurons') else [nrn_pop]
    return [soma_surface_area(n) for n in nrns]


def soma_radii(nrn_pop):
    ''' Get the radii of the somata of a population of neurons

    Note:
        If a single neuron is passed, a single element list with the
        radius of its soma member is returned.
    '''
    nrns = nrn_pop.neurons if hasattr(nrn_pop, 'neurons') else [nrn_pop]
    return [n.soma.radius for n in nrns]


def trunk_section_lengths(nrn, neurite_type=NeuriteType.all):
    '''list of lengths of trunk sections of neurites in a neuron'''
    neurite_filter = is_type(neurite_type)
    return [mm.section_length(s.root_node.points) for s in nrn.neurites if neurite_filter(s)]


def trunk_origin_radii(nrn, neurite_type=NeuriteType.all):
    ''' list of lengths of trunk sections of neurites in a neuron'''
    neurite_filter = is_type(neurite_type)
    return [s.root_node.points[0][COLS.R] for s in nrn.neurites if neurite_filter(s)]


def trunk_origin_azimuths(nrn, neurite_type=NeuriteType.all):
    '''Get a list of all the trunk origin azimuths of a neuron or population

    The azimuth is defined as Angle between x-axis and the vector
    defined by (initial tree point - soma center) on the x-z plane.

    The range of the azimuth angle [-pi, pi] radians
    '''
    neurite_filter = is_type(neurite_type)
    nrns = nrn.neurons if hasattr(nrn, 'neurons') else [nrn]

    def _azimuth(section, soma):
        '''Azimuth of a section'''
        vector = mm.vector(section[0], soma.center)
        return np.arctan2(vector[COLS.Z], vector[COLS.X])

    return [_azimuth(s.root_node.points, n.soma)
            for n in nrns for s in n.neurites if neurite_filter(s)]


def trunk_origin_elevations(nrn, neurite_type=NeuriteType.all):
    '''Get a list of all the trunk origin elevations of a neuron or population

    The elevation is defined as the angle between x-axis and the
    vector defined by (initial tree point - soma center)
    on the x-y half-plane.

    The range of the elevation angle [-pi/2, pi/2] radians
    '''
    neurite_filter = is_type(neurite_type)
    nrns = nrn.neurons if hasattr(nrn, 'neurons') else [nrn]

    def _elevation(section, soma):
        '''Elevation of a section'''
        vector = mm.vector(section[0], soma.center)
        norm_vector = np.linalg.norm(vector)

        if norm_vector >= np.finfo(type(norm_vector)).eps:
            return np.arcsin(vector[COLS.Y] / norm_vector)
        else:
            raise ValueError("Norm of vector between soma center and section is almost zero.")

    return [_elevation(s.root_node.points, n.soma)
            for n in nrns for s in n.neurites if neurite_filter(s)]
