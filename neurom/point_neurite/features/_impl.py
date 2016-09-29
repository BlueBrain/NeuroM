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

''' Neurite Related Features'''

from functools import wraps
from functools import partial
from .. import point_tree as _tr
from ..core import iter_neurites
from neurom.core.types import NeuriteType
from neurom.point_neurite.core import Neuron
from neurom.point_neurite import treefunc as _mt
from neurom import morphmath as _mm
from neurom.core.types import tree_type_checker as _ttc
from neurom.point_neurite import segments as _seg
from neurom.point_neurite import sections as _sec
from neurom.point_neurite import bifurcations as _bifs
from neurom.point_neurite import points as _pts
from neurom.morphmath import sphere_area


def feature_getter(mapfun):
    ''' Wrapper around already existing feature functions
    '''
    def wrapped(neurites, neurite_type=NeuriteType.all):
        '''Extracts feature from an object with neurites, i.e. either neurite, neuron, or population
        '''
        return iter_neurites(neurites, mapfun, _ttc(neurite_type))

    return wrapped


def count(f):
    ''' Counts the output of the wrapper wrapper.
    '''
    @wraps(f)
    def wrapped(neurites, neurite_type=NeuriteType.all):
        ''' placeholderg'''
        yield sum(1 for _ in f(neurites, neurite_type))

    return wrapped


def sum_feature(f):
    ''' Counts the output of the wrapper wrapper.
    '''
    @wraps(f)
    def wrapped(neurites, neurite_type=NeuriteType.all):
        ''' yields the sum of the function'''
        yield sum(f(neurites, neurite_type))

    return wrapped


section_lengths = feature_getter(_sec.length)
section_areas = feature_getter(_sec.area)
section_volumes = feature_getter(_sec.volume)
section_branch_orders = feature_getter(_sec.branch_order)
number_of_sections = count(feature_getter(_sec.identity))

segment_lengths = feature_getter(_seg.length)
number_of_segments = count(feature_getter(_seg.identity))
segment_taper_rates = feature_getter(_seg.taper_rate)
segment_radii = feature_getter(_seg.radius)
segment_x_coordinates = feature_getter(_seg.x_coordinate)
segment_y_coordinates = feature_getter(_seg.y_coordinate)
segment_z_coordinates = feature_getter(_seg.z_coordinate)

local_bifurcation_angles = feature_getter(_bifs.local_angle)
remote_bifurcation_angles = feature_getter(_bifs.remote_angle)
bifurcation_number = count(feature_getter(_bifs.identity))
partition = feature_getter(_bifs.partition)


def total_length_per_neurite(neurons, neurite_type=NeuriteType.all):
    '''Get an iterable with the total length of a neurite for a given neurite type'''
    return (sum(_sec.length(ss) for ss in _tr.isection(n))
            for n in iter_neurites(neurons, filt=_ttc(neurite_type)))

total_length = sum_feature(total_length_per_neurite)


def neurite_number(neurons, neurite_type=NeuriteType.all):
    '''Get an iterable with the number of neurites for a given neurite type
    '''
    yield sum(1 for n in iter_neurites(neurons, filt=_ttc(neurite_type)))


def number_of_sections_per_neurite(neurons, neurite_type=NeuriteType.all):
    '''Get an iterable with the number of sections for a given neurite type'''
    return (_sec.count(n) for n in iter_neurites(neurons, filt=_ttc(neurite_type)))


def section_path_distances(neurites, use_start_point=False, neurite_type=NeuriteType.all):
    '''
    Get section path distances of all neurites of a given type
    The section path distance is measured to the neurite's root.

    Parameters:
        use_start_point: boolean\
            if true, use the section's first point,\
            otherwise use the end-point (default False)
        neurite_type: NeuriteType\
            Type of neurites to be considered (default all)

    Returns:
        Iterable containing the section path distances.

    '''
    magic_iter = (_sec.start_point_path_length if use_start_point
                  else _sec.end_point_path_length)
    return iter_neurites(neurites, magic_iter, _ttc(neurite_type))


def segment_radial_distances(neurites, origin=None, neurite_type=NeuriteType.all):
    '''Get an iterable containing section radial distances to origin of\
        all neurites of a given type

    Parameters:
        origin: Point wrt which radial dirtance is calulated\
            (default tree root)
        use_start_point: if true, use the section's first point,\
            otherwise use the end-point (default False)
        neurite_type: Type of neurites to be considered (default all)

    '''

    def i_segment_radial_dist(tree):
        '''Return an iterator of radial distances of tree segments

        The radial distance is the euclidian distance between the either the
        middle point of the segment and the first node of the tree.

        Parameters:
            tree: tree object
        '''
        pos = tree.value if origin is None else origin
        return _tr.imap_val(lambda s: _mm.segment_radial_dist(s, pos), _tr.isegment(tree))

    def f(n):
        '''neurite identity function'''
        return n

    f.iter_type = i_segment_radial_dist
    return iter_neurites(neurites, f, _ttc(neurite_type))


def section_radial_distances(neurites, origin=None, use_start_point=False,
                             neurite_type=NeuriteType.all):
    '''Get an iterable containing section radial distances to origin of\
        all neurites of a given type

    Parameters:
        origin: Point wrt which radial dirtance is calulated\
            (default tree root)
        use_start_point: if true, use the section's first point,\
            otherwise use the end-point (default False)
        neurite_type: Type of neurites to be considered (default all)

    '''
    def f(n):
        '''neurite identity function'''
        return n
    f.iter_type = partial(_mt.i_section_radial_dist,
                          pos=origin, use_start_point=use_start_point)
    return iter_neurites(neurites, f, _ttc(neurite_type))


def trunk_section_lengths(neurons, neurite_type=NeuriteType.all):
    '''Get the trunk section lengths of a given type in a neuron'''
    return (_mt.trunk_section_length(t)
            for t in iter_neurites(neurons, filt=_ttc(neurite_type)))


def trunk_origin_radii(neurons, neurite_type=NeuriteType.all):
    '''Get the trunk origin radii of a given type in a neuron'''
    return (_pts.radius(t) for t in iter_neurites(neurons, filt=_ttc(neurite_type)))


def principal_directions_extents(neurons, neurite_type=NeuriteType.all, direction='first'):
    ''' Get principal direction extent of either a neurite or the total neurites
    from a neuron or a population.

    Parameters:
        direction: string \
            it can be either 'first', 'second' or 'third' \
            corresponding to the respective principal direction \
            of the extent

    Returns:
        Iterator containing the extents of the input neurites

    '''
    n = 0 if direction == 'first' else (1 if direction == 'second' else 2)
    return (_mt.principal_direction_extent(t)[n]
            for t in iter_neurites(neurons, filt=_ttc(neurite_type)))


def as_neuron_list(func):
    ''' If a single neuron is provided to the function it passes the argument as a list of a single
    element. If a population is passed as an argument, it replaces it by its neurons.
    '''
    @wraps(func)
    def wrapped(obj, *args, **kwargs):
        ''' Takes care of the neuron feature input. By using this decorator the neuron functions
        can take as an input a single neuron, list of neurons or a population.
        '''
        neurons = [obj] if isinstance(obj, Neuron) else (obj.neurons if hasattr(obj, 'neurons')
                                                         else obj)
        return func(neurons, *args, **kwargs)
    return wrapped


@as_neuron_list
def soma_radii(neurons):
    '''Get the radius of the soma'''
    return (nrn.soma.radius for nrn in neurons)


@as_neuron_list
def soma_surface_areas(neurons):
    '''Get the surface area of the soma.

    Note:
        The surface area is calculated by assuming the soma is spherical.
    '''
    return (sphere_area(nrn.soma.radius) for nrn in neurons)


@as_neuron_list
def trunk_origin_azimuths(neurons, neurite_type=NeuriteType.all):
    '''Applies the trunk_origin_azimuth function on the soma and the neurites of each
    neuron.
    '''
    for nrn in neurons:
        for neu in nrn.neurites:
            if _ttc(neurite_type)(neu):
                yield _mt.trunk_origin_azimuth(neu, nrn.soma)


@as_neuron_list
def trunk_origin_elevations(neurons, neurite_type=NeuriteType.all):
    '''Applies the trunk_origin_elevation function on the soma and the neurites of each neuron.
    '''
    for nrn in neurons:
        for neu in nrn.neurites:
            if _ttc(neurite_type)(neu):
                yield _mt.trunk_origin_elevation(neu, nrn.soma)
