
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

'''Working Ensemble for a general get_feature functionality'''

from enum import Enum, unique
import os
from neurom.io.utils import get_morph_files, load_neuron as _load
from neurom.analysis.morphtree import set_tree_type as _set_tt
from neurom.core.types import TreeType
from neurom.core.types import checkTreeType
from neurom.core.tree import ipreorder
from neurom.core.tree import isection
from neurom.core.tree import isegment
from neurom.core.neuron import Neuron as CoreNeuron
from neurom.analysis.morphmath import section_length
from neurom.analysis.morphmath import segment_length
from neurom.analysis.morphtree import i_local_bifurcation_angle
from neurom.analysis.morphtree import i_remote_bifurcation_angle
from neurom.analysis.morphtree import i_section_radial_dist
from neurom.analysis.morphtree import i_section_path_length
from neurom.analysis.morphtree import n_sections
import math
import numpy as np
from neurom.core.population import Population as CorePopulation
from itertools import chain


@unique
class Feature(Enum):
    '''Enum representing valid features'''
    soma_radius = 1
    soma_surface_area = 2
    segment_lengths = 3
    section_lengths = 4
    local_bifurcation_angles = 5
    remote_bifurcation_angles = 6
    section_path_distances = 7
    section_radial_distances = 8
    n_neurites = 9
    n_sections = 10
    n_sections_per_neurite = 11


FeatureDepth = {
    Feature.soma_radius: 0,
    Feature.soma_surface_area: 0,
    Feature.n_neurites: 0,
    Feature.n_sections: 0,
    Feature.segment_lengths: 1,
    Feature.section_lengths: 1,
    Feature.local_bifurcation_angles: 1,
    Feature.remote_bifurcation_angles: 1,
    Feature.section_path_distances: 1,
    Feature.section_radial_distances: 1,
    Feature.n_sections_per_neurite: 1
}


def _check_valid_kwargs(kwargs, allowed_kwargs):
    ''''Check if kwarg is in allowed kwargs'''
    assert all([kwarg in allowed_kwargs for kwarg in kwargs])


def load_neuron(filename):
    '''Load a Neuron from a file'''
    return Neuron(_load(filename, _set_tt))


def load_neurons(directory):
    '''Create a list of Neuron objects from each morphology file in directory'''
    return [load_neuron(m) for m in get_morph_files(directory)]


def load_population(directory):
    '''Create a population object from all morphologies in a directory'''
    pop = Population(load_neurons(directory))
    pop.name = os.path.basename(directory)
    return pop


class Neuron(CoreNeuron):
    ''' Modified Neuron Class in order to have get_feature functionality

    '''

    def __init__(self, neuron, iterable_type=np.array):
        super(Neuron, self).__init__(neuron.soma, neuron.neurites, neuron.name)
        self._iterable_type = iterable_type
        self._fmap = self._setup_function_map()

    def _setup_function_map(self):
        '''Generate function map for feature acquisition'''
        return {Feature.soma_radius:
                {
                    'func': getattr,
                    'args': (self.soma, 'radius'),
                    'valid_kwargs': []
                },
                Feature.soma_surface_area:
                {
                    'func': self._soma_surface,
                    'args': (),
                    'valid_kwargs': []
                },
                Feature.segment_lengths:
                {
                    'func': self.iter_segments,
                    'args': (segment_length,),
                    'valid_kwargs': ['neurite_type']
                },
                Feature.section_lengths:
                {
                    'func': self.iter_sections,
                    'args': (section_length,),
                    'valid_kwargs': ['neurite_type']
                },
                Feature.local_bifurcation_angles:
                {
                    'func': self.neurite_loop,
                    'args': (i_local_bifurcation_angle,),
                    'valid_kwargs': []
                },
                Feature.remote_bifurcation_angles:
                {
                    'func': self.neurite_loop,
                    'args': (i_remote_bifurcation_angle,),
                    'valid_kwargs': []
                },
                Feature.section_path_distances:
                {
                    'func': self.neurite_loop,
                    'args': (),
                    'valid_kwargs': ['neurite_type', 'use_start_point']
                },
                Feature.section_radial_distances:
                {
                    'func': self._section_radial_distances,
                    'args': (),
                    'valid_kwargs': ['origin', 'use_start_point', 'neurite_type']
                },
                Feature.n_neurites:
                {
                    'func': self._n_neurites,
                    'args': (),
                    'valid_kwargs': ['neurite_type']
                },
                Feature.n_sections:
                {
                    'func': self._n_sections,
                    'args': (),
                    'valid_kwargs': ['neurite_type']
                },
                Feature.n_sections_per_neurite:
                {
                    'func': self._n_sections_per_neurite,
                    'args': (),
                    'valid_kwargs': ['neurite_type']
                }}

    def get_feature(self, feature, **kwargs):
        '''Get the input feature'''
        depth = FeatureDepth[feature]

        func = self._fmap[feature]['func']

        args = self._fmap[feature]['args']

        _check_valid_kwargs(kwargs, self._fmap[feature]['valid_kwargs'])

        out = func(*args, **kwargs)

        return self._pkg(out) if depth > 0 else out

    def _soma_surface(self):
        '''Get the surface area of the soma.

        Note:
            The surface area is calculated by assuming the soma is spherical.
        '''
        return 4 * math.pi * self.soma.radius ** 2

    def _section_path_distances(self, **kwargs):
        '''
        Get section path distances of all neurites of a given type
        The section path distance is measured to the neurite's root.

        Parameters:
            use_start_point: boolean\
            if true, use the section's first point,\
            otherwise use the end-point (default False)
            neurite_type: TreeType\
            Type of neurites to be considered (default all)

        Returns:
            Iterable containing the section path distances.
        '''

        use_start_point = kwargs.get('use_start_point', False)

        neurite_type = kwargs.get('neurite_type', TreeType.all)

        return self.neurite_loop(lambda t: i_section_path_length(t, use_start_point),
                                 neurite_type=neurite_type)

    def _section_radial_distances(self, **kwargs):
        '''Get an iterable containing section radial distances to origin of\
           all neurites of a given type

        Parameters:
            origin: Point wrt which radial dirtance is calulated\
                    (default tree root)
            use_start_point: if true, use the section's first point,\
                             otherwise use the end-point (default False)
            neurite_type: Type of neurites to be considered (default all)
        '''
        origin = kwargs.get('origin', None)
        use_start_point = kwargs.get('use_start_point', False)
        neurite_type = kwargs.get('', TreeType.all)

        return self.neurite_loop(lambda t: i_section_radial_dist(t, origin,
                                                                 use_start_point),
                                 neurite_type=neurite_type)

    def _n_sections(self, **kwargs):
        '''Get the number of sections of a given type'''
        neurite_type = kwargs.get('neurite_type', TreeType.all)
        return sum(n_sections(t) for t in self.neurites if checkTreeType(neurite_type, t.type))

    def _n_neurites(self, **kwargs):
        '''Get the number of neurites of a given type in a neuron'''
        neurite_type = kwargs.get('neurite_type', TreeType.all)
        return sum(1 for n in self.neurites if checkTreeType(neurite_type, n.type))

    def _n_sections_per_neurite(self, **kwargs):
        '''Get an iterable with the number of sections for a given neurite type'''
        neurite_type = kwargs.get('neurite_type', TreeType.all)
        return self._iterable_type(
            [n_sections(n) for n in self.neurites
             if checkTreeType(neurite_type, n.type)]
        )

    def iter_neurites(self, iterator_type, mapping=None, neurite_type=TreeType.all):
        '''Iterate over collection of neurites applying iterator_type

        Parameters:
            iterator_type: Type of iterator with which to perform the iteration.\
            (e.g. isegment, isection, i_section_path_length)
            mapping: mapping function to be applied to the target of iteration.\
            (e.g. segment_length). Must be compatible with the iterator_type.
            neurite_type: TreeType object. Neurites of incompatible type are\
            filtered out.

        Returns:
            Iterator of mapped iteration targets.

        Example:
            Get the total volume of all neurites in the cell and the total\
                length or neurites from their segments.

        >>> from neurom import ezy
        >>> from neurom.analysis import morphmath as mm
        >>> from neurom.core import tree as tr
        >>> nrn = ezy.load_neuron('test_data/swc/Neuron.swc')
        >>> v = sum(nrn.iter_neurites(tr.isegment, mm.segment_volume))
        >>> tl = sum(nrn.iter_neurites(tr.isegment, mm.segment_length)))

        '''
        return self.i_neurites(iterator_type,
                               mapping,
                               tree_filter=lambda t: checkTreeType(neurite_type,
                                                                   t.type))

    def iter_points(self, mapfun, neurite_type=TreeType.all):
        '''Iterator to neurite points with mapping

        Parameters:
            mapfun: mapping function to be applied to points.
            neurite_type: type of neurites to iterate over.
        '''
        return self.iter_neurites(ipreorder, mapfun, neurite_type)

    def iter_segments(self, mapfun, neurite_type=TreeType.all, **kwargs):
        '''Iterator to neurite segments with mapping

        Parameters:
            mapfun: mapping function to be applied to segments.
            neurite_type: type of neurites to iterate over.
        '''
        neurite_type = kwargs.get('neurite_type', TreeType.all)

        return self.iter_neurites(isegment, mapfun, neurite_type)

    def iter_sections(self, mapfun, neurite_type=TreeType.all):
        '''Iterator to neurite sections with mapping

        Parameters:
            mapfun: mapping function to be applied to sections.
            neurite_type: type of neurites to iterate over.
        '''
        return self.iter_neurites(isection, mapfun, neurite_type)

    def neurite_loop(self, iterator_type, mapping=None, neurite_type=TreeType.all):
        '''Iterate over collection of neurites applying iterator_type

        Parameters:
            iterator_type: Type of iterator with which to perform the iteration.
            (e.g. isegment, isection, i_section_path_length)
            mapping: mapping function to be applied to the target of iteration.
            (e.g. segment_length). Must be compatible with the iterator_type.
            neurite_type: TreeType object. Neurites of incompatible type are
            filtered out.

        Returns:
            Iterable containing the iteration targets after mapping.
        '''
        return self._pkg(self.iter_neurites(iterator_type, mapping, neurite_type))

    def _pkg(self, iterator):
        '''Create an iterable from an iterator'''
        return self._iterable_type([i for i in iterator])


class Population(CorePopulation):
    '''Population Class

    Arguments:
        neurons: list of neurons (core or ezy)
    '''

    def __init__(self, neurons):
        super(Population, self).__init__(neurons)

    def get_feature(self, feature, **kwargs):
        '''Get Feature
        '''
        depth = FeatureDepth[feature]

        values = [neu.get_feature(feature, **kwargs) for neu in self.neurons]

        return list(chain(*values)) if depth > 0 else values

    def iter_somata(self):
        '''
        Iterate over the neuron somata

            Returns:
                Iterator of neuron somata
        '''
        return iter(self.somata)

    def get_n_neurites(self, neurite_type=TreeType.all):
        '''Get the number of neurites of a given type in a population'''
        return sum(nrn.get_n_neurites(neurite_type=neurite_type) for nrn in self.iter_neurons())

    def iter_neurites(self):
        '''
        Iterate over the neurites

            Returns:
                Iterator of neurite tree iterators
        '''
        return iter(self.neurites)

    def iter_neurons(self):
        '''
        Iterate over the neurons in the population

            Returns:
                Iterator of neurons
        '''
        return iter(self.neurons)
