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
from neurom.analysis.morphtree import set_tree_type
from neurom.analysis.morphtree import i_section_length
from neurom.analysis.morphtree import i_segment_length
from neurom.analysis.morphtree import i_local_bifurcation_angle
from neurom.analysis.morphtree import i_remote_bifurcation_angle
from neurom.analysis.morphtree import i_section_radial_dist
from neurom.analysis.morphtree import i_section_path_length
from neurom.analysis.morphtree import n_sections
from neurom.view import view
import numpy as np
from itertools import chain


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

    def get_section_lengths(self, neurite_type=TreeType.all):
        '''Get an iterable containing the lengths of all sections of a given type'''
        return self._neurite_loop(neurite_type, i_section_length)

    def get_segment_lengths(self, neurite_type=TreeType.all):
        '''Get an iterable containing the lengths of all segments of a given type'''
        return self._neurite_loop(neurite_type, i_segment_length)

    def get_local_bifurcation_angles(self, neurite_type=TreeType.all):
        '''Get local bifircation angles of all segments of a given type

        The local bifurcation angle is defined as the angle between
        the first segments of the bifurcation branches.

        Returns:
            Iterable containing bifurcation angles in radians
        '''
        return self._neurite_loop(neurite_type, i_local_bifurcation_angle)

    def get_remote_bifurcation_angles(self, neurite_type=TreeType.all):
        '''Get remote bifircation angles of all segments of a given type

        The remote bifurcation angle is defined as the angle between
        the lines joining the bifurcation point to the last points
        of the bifurcated sections.

        Returns:
            Iterable containing bifurcation angles in radians
        '''
        return self._neurite_loop(neurite_type, i_remote_bifurcation_angle)

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
        return self._neurite_loop(neurite_type,
                                  lambda t: i_section_radial_dist(t, origin,
                                                                  use_start_point))

    def get_section_path_lengths(self, use_start_point=False,
                                 neurite_type=TreeType.all):
        '''Get section path lengths of all neurites of a given type

        The section path length is measured to the neurite's root.

        Parameters:
            use_start_point: if true, use the section's first point,\
                             otherwise use the end-point (default False)
            neurite_type: Type of neurites to be considered (default all)

        Returns:
            Iterable containing the section path lengths.
        '''
        return self._neurite_loop(neurite_type,
                                  lambda t: i_section_path_length(t, use_start_point))

    def get_n_sections(self, neurite_type=TreeType.all):
        '''Get the number of sections of a given type'''
        return sum([n_sections(t) for t in self._filter_neurites(neurite_type)])

    def get_n_sections_per_neurite(self, neurite_type=TreeType.all):
        '''Get an iterable with the number of sections for a given neurite type'''
        neurites = self._filter_neurites(neurite_type)
        return self._iterable_type([n_sections(n) for n in neurites])

    def get_n_neurites(self, neurite_type=TreeType.all):
        '''Get the number of neurites of a given type in a neuron'''
        return len(self._filter_neurites(neurite_type))

    def _neurite_loop(self, neurite_type, iterator_type):
        '''Iterate over collection of neurites applying iterator_type
        '''
        neurites = self._filter_neurites(neurite_type)
        l = [[i for i in iterator_type(t)] for t in neurites]
        return self._iterable_type([i for i in chain(*l)])

    def _filter_neurites(self, neurite_type):
        '''Filter neurites by type'''
        return self._nrn.neurite_trees if (neurite_type is TreeType.all) else(
            [t for t in self._nrn.neurite_trees if t.type is neurite_type])

    def plot(self, *args, **kwargs):
        '''Make a 2D plot of this neuron

        Forwards arguments to neurom.view.view.neuron()
        '''
        return view.neuron(self._nrn, *args, **kwargs)

    def plot3d(self, *args, **kwargs):
        '''Make a 3D plot of this neuron

        Forwards arguments to neurom.view.view.neuron3d()
        '''
        return view.neuron3d(self._nrn, *args, **kwargs)
