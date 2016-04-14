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


from itertools import product
from neurom.core.types import NeuriteType
from neurom.core.types import tree_type_checker
from neurom.utils import deprecated
from neurom import segments as _seg
from neurom import sections as _sec
from neurom import bifurcations as _bifs
from neurom import points as _pts
from neurom import iter_neurites
from neurom.core.neuron import Neuron as CoreNeuron
from neurom.analysis.morphtree import i_section_radial_dist
from neurom.analysis.morphtree import trunk_section_length
from neurom.analysis.morphtree import compare_trees
import math
import numpy as np


class Neuron(CoreNeuron):
    '''Class with basic analysis and plotting functionality

    By default returns iterables as numpy.arrays and applies no filtering.

    Arguments:
        neuron: neuron-like object
        iterable_type: type of iterable to return from methods returning \
            collections (e.g list, tuple, numpy.array).

    Raises:
        neurom.exceptions.SomaError if no soma can be built from the data
        neurom.exceptions.IDSequenceError if ID sequence invalid

    Example:
        get the segment lengths of all apical dendrites in a neuron morphology.

    >>> from neurom import ezy
    >>> nrn = ezy.load_neuron('test_data/swc/Neuron.swc')
    >>> nrn.get_segment_lengths(ezy.NeuriteType.apical_dendrite)

    Example:
        use lists instead of numpy arrays and get section \
    lengths for the axon. Read an HDF5 v1 file:

    >>> from neurom import ezy
    >>> nrn = ezy.load_neuron('test_data/h5/v1/Neuron.h5', iterable_type=list)
    >>> nrn.get_section_lengths(ezy.NeuriteType.axon)

    Example:
        plot the neuron and save to a file

    >>> fig, ax = nrn.plot()
    >>> fig.show()
    >>> fig.savefig('nrn.png')

    '''

    def __init__(self, neuron, iterable_type=np.array):
        super(Neuron, self).__init__(neuron.soma, neuron.neurites, neuron.name)
        self._iterable_type = iterable_type

    def __eq__(self, other):
        return False if not isinstance(self, type(other)) else \
               all(self._compare_neurites(other, ttype) for ttype in
                   [NeuriteType.axon, NeuriteType.basal_dendrite,
                    NeuriteType.apical_dendrite, NeuriteType.undefined])

    @deprecated('Use ezy.get instead.')
    def get_section_lengths(self, neurite_type=NeuriteType.all):
        '''Get an iterable containing the lengths of all sections of a given type'''
        return self._pkg(_sec.length, neurite_type)

    @deprecated('Use ezy.get instead.')
    def get_segment_lengths(self, neurite_type=NeuriteType.all):
        '''Get an iterable containing the lengths of all segments of a given type'''
        return self._pkg(_seg.length, neurite_type)

    @deprecated('Use ezy.get instead.')
    def get_soma_radius(self):
        '''Get the radius of the soma'''
        return self.soma.radius

    @deprecated('Use ezy.get instead.')
    def get_soma_surface_area(self):
        '''Get the surface area of the soma.

        Note:
            The surface area is calculated by assuming the soma is spherical.
        '''
        return 4 * math.pi * self.get_soma_radius() ** 2

    @deprecated('Use ezy.get instead.')
    def get_local_bifurcation_angles(self, neurite_type=NeuriteType.all):
        '''Get local bifircation angles of all segments of a given type

        The local bifurcation angle is defined as the angle between
        the first segments of the bifurcation branches.

        Returns:
            Iterable containing bifurcation angles in radians
        '''
        return self._pkg(_bifs.local_angle, neurite_type)

    @deprecated('Use ezy.get instead.')
    def get_remote_bifurcation_angles(self, neurite_type=NeuriteType.all):
        '''Get remote bifircation angles of all segments of a given type

        The remote bifurcation angle is defined as the angle between
        the lines joining the bifurcation point to the last points
        of the bifurcated sections.

        Returns:
            Iterable containing bifurcation angles in radians
        '''
        return self._pkg(_bifs.remote_angle, neurite_type)

    @deprecated('Use ezy.get instead.')
    def get_section_radial_distances(self, origin=None, use_start_point=False,
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
        return self._neurite_loop(lambda t: i_section_radial_dist(t, origin,
                                                                  use_start_point),
                                  neurite_type=neurite_type)

    @deprecated('Use ezy.get instead.')
    def get_section_path_distances(self, use_start_point=False,
                                   neurite_type=NeuriteType.all):
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
        return self._pkg(magic_iter, neurite_type)

    @deprecated('Use ezy.get instead.')
    def get_n_sections(self, neurite_type=NeuriteType.all):
        '''Get the number of sections of a given type'''
        tree_filter = tree_type_checker(neurite_type)
        return _sec.count(self, tree_filter)

    @deprecated('Use ezy.get instead.')
    def get_n_sections_per_neurite(self, neurite_type=NeuriteType.all):
        '''Get an iterable with the number of sections for a given neurite type'''
        tree_filter = tree_type_checker(neurite_type)
        return self._iterable_type(
            [_sec.count(n) for n in self.neurites if tree_filter(n)]
        )

    @deprecated('Use ezy.get instead.')
    def get_n_neurites(self, neurite_type=NeuriteType.all):
        '''Get the number of neurites of a given type in a neuron'''
        tree_filter = tree_type_checker(neurite_type)
        return sum(1 for n in self.neurites if tree_filter(n))

    @deprecated('Use ezy.get instead.')
    def get_trunk_origin_radii(self, neurite_type=NeuriteType.all):
        '''Get the trunk origin radii of a given type in a neuron'''
        tree_filter = tree_type_checker(neurite_type)
        return self._iterable_type(
            [_pts.radius(t) for t in self.neurites if tree_filter(t)]
        )

    @deprecated('Use ezy.get instead.')
    def get_trunk_section_lengths(self, neurite_type=NeuriteType.all):
        '''Get the trunk section lengths of a given type in a neuron'''
        tree_filter = tree_type_checker(neurite_type)
        return self._iterable_type(
            [trunk_section_length(t) for t in self.neurites if tree_filter(t)]
        )

    def _iter_neurites(self, iterator_type, mapping=None, neurite_type=NeuriteType.all):
        '''Iterate over collection of neurites applying iterator_type

        Parameters:
            iterator_type: Type of iterator with which to perform the iteration.\
            (e.g. isegment, isection)
            mapping: mapping function to be applied to the target of iteration.\
            (e.g. segment_length). Must be compatible with the iterator_type.
            neurite_type: NeuriteType object. Neurites of incompatible type are\
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
                               tree_filter=tree_type_checker(neurite_type))

    def _pkg(self, magic_iter, neurite_type=NeuriteType.all):
        '''Return an iterable built from magic_iter'''
        stuff = list(
            iter_neurites(self,
                          magic_iter,
                          tree_type_checker(neurite_type))
        )
        return self._iterable_type(stuff)

    def _neurite_loop(self, iterator_type, mapping=None, neurite_type=NeuriteType.all):
        '''Iterate over collection of neurites applying iterator_type

        Parameters:
            iterator_type: Type of iterator with which to perform the iteration.
            (e.g. isegment, isection)
            mapping: mapping function to be applied to the target of iteration.
            (e.g. segment_length). Must be compatible with the iterator_type.
            neurite_type: NeuriteType object. Neurites of incompatible type are
            filtered out.

        Returns:
            Iterable containing the iteration targets after mapping.
        '''
        return self._iterable_type(
            list(self._iter_neurites(iterator_type, mapping, neurite_type))
        )

    def _compare_neurites(self, other, neurite_type, comp_function=compare_trees):
        '''
        Find the identical pair of neurites of determined type if existent.

        Returns:
            False if pair does not exist or not identical. True otherwise.
        '''
        neurites1 = [neu for neu in self.neurites if neu.type == neurite_type]

        neurites2 = [neu for neu in other.neurites if neu.type == neurite_type]

        if len(neurites1) == len(neurites2):

            return True if len(neurites1) == 0 and len(neurites2) == 0 else \
                   len(neurites1) - sum(1 for neu1, neu2 in
                                        product(neurites1, neurites2)
                                        if comp_function(neu1, neu2)) == 0
        else:

            return False
