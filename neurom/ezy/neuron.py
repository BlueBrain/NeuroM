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

''' Neuron class with basic analysis and plotting capabilities. '''

from neurom.io.utils import load_neuron
from neurom.core.types import TreeType
from neurom.core.types import checkTreeType
from neurom.core.tree import ipreorder
from neurom.core.tree import isection
from neurom.core.tree import isegment
from neurom.core.tree import i_chain as i_neurites
from neurom.analysis.morphmath import section_length
from neurom.analysis.morphmath import segment_length
from neurom.analysis.morphtree import set_tree_type
from neurom.analysis.morphtree import i_local_bifurcation_angle
from neurom.analysis.morphtree import i_remote_bifurcation_angle
from neurom.analysis.morphtree import i_section_radial_dist
from neurom.analysis.morphtree import i_section_path_length
from neurom.analysis.morphtree import n_sections
import math
import numpy as np


class Neuron(object):
    '''Class with basic analysis and plotting functionality

    By default returns iterables as numpy.arrays and applies no filtering.

    Arguments:
        filename: path to morphology file to be loaded.
        iterable_type: type of iterable to return from methods returning \
            collections (e.g list, tuple, numpy.array).

    Raises:
        neurom.exceptions.SomaError if no soma can be built from the data
        neurom.exceptions.NonConsecutiveIDsError if data IDs not consecutive

    Example:
        get the segment lengths of all apical dendrites in a neuron morphology.

    >>> from neurom import ezy
    >>> nrn = ezy.Neuron('test_data/swc/Neuron.swc')
    >>> nrn.get_segment_lengths(ezy.TreeType.apical_dendrite)

    Example:
        iterate over the segment surface areas of all axons in a neuron\
            morphology.

    >>> from neurom import ezy
    >>> from neurom.analysis import morphmath as mm
    >>> nrn = ezy.Neuron('test_data/swc/Neuron.swc')
    >>> for a in nrn.iter_segments(mm.segment_area, ezy.TreeType.axon):
          print (a)

    Example:
        calculate the mean volume of all neurite segments on a neuron\
            morphology

    >>> from neurom import ezy
    >>> from neurom.analysis import morphmath as mm
    >>> import numpy as np
    >>> nrn = ezy.Neuron('test_data/swc/Neuron.swc')
    >>> mv = np.mean([v for v in nrn.iter_segments(mm.segment_volume)])

    Example:
        use lists instead of numpy arrays and get section \
    lengths for the axon. Read an HDF5 v1 file:

    >>> from neurom import ezy
    >>> nrn = ezy.Neuron('test_data/h5/v1/Neuron.h5', iterable_type=list)
    >>> nrn.get_section_lengths(ezy.TreeType.axon)

    Example:
        plot the neuron and save to a file

    >>> fig, ax = nrn.plot()
    >>> fig.show()
    >>> fig.savefig('nrn.png')

    '''

    def __init__(self, filename, iterable_type=np.array):
        self._iterable_type = iterable_type
        self._nrn = load_neuron(filename, set_tree_type)
        self.soma = self._nrn.soma
        self.neurites = self._nrn.neurites

    def get_section_lengths(self, neurite_type=TreeType.all):
        '''Get an iterable containing the lengths of all sections of a given type'''
        return self._pkg(self.iter_sections(section_length, neurite_type))

    def get_segment_lengths(self, neurite_type=TreeType.all):
        '''Get an iterable containing the lengths of all segments of a given type'''
        return self._pkg(self.iter_segments(segment_length, neurite_type))

    def get_soma_radius(self):
        '''Get the radius of the soma'''
        return self._nrn.soma.radius

    def get_soma_surface_area(self):
        '''Get the surface area of the soma.

        Note:
            The surface area is calculated by assuming the soma is spherical.
        '''
        return 4 * math.pi * self.get_soma_radius() ** 2

    def get_local_bifurcation_angles(self, neurite_type=TreeType.all):
        '''Get local bifircation angles of all segments of a given type

        The local bifurcation angle is defined as the angle between
        the first segments of the bifurcation branches.

        Returns:
            Iterable containing bifurcation angles in radians
        '''
        return self.neurite_loop(i_local_bifurcation_angle,
                                 neurite_type=neurite_type)

    def get_remote_bifurcation_angles(self, neurite_type=TreeType.all):
        '''Get remote bifircation angles of all segments of a given type

        The remote bifurcation angle is defined as the angle between
        the lines joining the bifurcation point to the last points
        of the bifurcated sections.

        Returns:
            Iterable containing bifurcation angles in radians
        '''
        return self.neurite_loop(i_remote_bifurcation_angle,
                                 neurite_type=neurite_type)

    def get_section_radial_distances(self, origin=None, use_start_point=False,
                                     neurite_type=TreeType.all):
        '''Get an iterable containing section radial distances to origin of\
           all neurites of a given type

        Parameters:
            origin: Point wrt which radial dirtance is calulated\
                    (default tree root)
            use_start_point: if true, use the section's first point,\
                             otherwise use the end-point (default False)
            neurite_type: Type of neurites to be considered (default all)
        '''
        return self.neurite_loop(lambda t: i_section_radial_dist(t, origin,
                                                                 use_start_point),
                                 neurite_type=neurite_type)

    def get_section_path_distances(self, use_start_point=False,
                                   neurite_type=TreeType.all):
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
        return self.neurite_loop(lambda t: i_section_path_length(t, use_start_point),
                                 neurite_type=neurite_type)

    def get_n_sections(self, neurite_type=TreeType.all):
        '''Get the number of sections of a given type'''
        return sum(n_sections(t) for t in self._nrn.neurites
                   if checkTreeType(neurite_type, t.type))

    def get_n_sections_per_neurite(self, neurite_type=TreeType.all):
        '''Get an iterable with the number of sections for a given neurite type'''
        return self._iterable_type(
            [n_sections(n) for n in self._nrn.neurites
             if checkTreeType(neurite_type, n.type)]
        )

    def get_n_neurites(self, neurite_type=TreeType.all):
        '''Get the number of neurites of a given type in a neuron'''
        return sum(1 for n in self._nrn.neurites
                   if checkTreeType(neurite_type, n.type))

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
        >>> nrn = ezy.Neuron('test_data/swc/Neuron.swc')
        >>> v = sum(nrn.iter_neurites(tr.isegment, mm.segment_volume))
        >>> tl = sum(nrn.iter_neurites(tr.isegment, mm.segment_length)))

        '''
        return i_neurites(self._nrn.neurites,
                          iterator_type,
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

    def iter_segments(self, mapfun, neurite_type=TreeType.all):
        '''Iterator to neurite segments with mapping

        Parameters:
            mapfun: mapping function to be applied to segments.
            neurite_type: type of neurites to iterate over.
        '''
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
