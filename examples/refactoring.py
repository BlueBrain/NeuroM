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

'''Refactored Example to Minimize code multiplication'''

import neurom.core.tree as tr
from neurom.core.types import TreeType
from neurom.core.dataformat import COLS
import neurom.analysis.morphtree as mtr
from neurom.core.types import checkTreeType

import numpy as np
from itertools import chain, imap


def imap_val(f, tree_iterator):
    '''Map function f to value of tree_iterator's target
    '''
    return imap(f, tr.val_iter(tree_iterator))


def i_chain(neurites, iterator_type, mapping=None, tree_filter=None):
    '''Returns a mapped iterator to a collection of trees

    Provides access to all the elements of all the trees
    in one iteration sequence.

    Parameters:
        trees: iterator or iterable of tree objects
        iterator_type: type of the iteration (segment, section, triplet...)
        mapping: optional function to apply to the iterator's target.
        tree_filter: optional top level filter on properties of tree objects.
    '''
    nrt = (neu.tree for neu in (neurites if tree_filter is None
                                else filter(tree_filter, neurites)))

    chain_it = chain(*imap(iterator_type, nrt))

    return chain_it if mapping is None else imap_val(mapping, chain_it)


class NeuriteFeatures(object):
    ''' NeuriteFeatures Class
    '''
    def __init__(self, neurites, iterable_type=np.array):
        ''' Construct a NeuriteFeatures object
        '''
        self._neurites = neurites
        self._iterable_type = iterable_type

    def _pkg(self, iterator):
        '''Create an iterable from an iterator'''
        return self._iterable_type([i for i in iterator])

    def _i_neurites(self, iterator_type, mapping=None, tree_filter=None):
        '''Returns a mapped iterator to all the neuron's neurites

        Provides access to all the elements of all the neurites
        in one iteration sequence.

        Parameters:
            iterator_type: type of the iteration (segment, section, triplet...)
            mapping: optional function to apply to the iterator's target.
            tree_filter: optional top level filter on properties of neurite tree objects.
        '''
        return i_chain(self._neurites, iterator_type, mapping, tree_filter)

    def _iter_neurites(self, iterator_type, mapping=None, neurite_type=TreeType.all):
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
        return self._i_neurites(iterator_type, mapping=mapping,
                                tree_filter=lambda t: checkTreeType(neurite_type,
                                                                    t.type))

    def _neurite_loop(self, iterator_type, mapping=None, neurite_type=TreeType.all):
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
        return self._pkg(self._iter_neurites(iterator_type, mapping, neurite_type))

    def local_bifurcation_angles(self, neurite_type=TreeType.all):
        '''Get local bifircation angles of all segments of a given type

        The local bifurcation angle is defined as the angle between
        the first segments of the bifurcation branches.

        Returns:
            Iterable containing bifurcation angles in radians
        '''
        return self._neurite_loop(mtr.i_local_bifurcation_angle,
                                  neurite_type=neurite_type)

    def n_sections(self, neurite_type=TreeType.all):
        '''Get the number of sections of a given type'''
        return sum(mtr.n_sections(neu.tree) for neu in self._neurites
                   if checkTreeType(neurite_type, neu.type))

    def remote_bifurcation_angles(self, neurite_type=TreeType.all):
        '''Get remote bifircation angles of all segments of a given type

        The remote bifurcation angle is defined as the angle between
        the lines joining the bifurcation point to the last points
        of the bifurcated sections.

        Returns:
            Iterable containing bifurcation angles in radians
        '''
        return self._neurite_loop(mtr.i_remote_bifurcation_angle,
                                  neurite_type=neurite_type)

    def section_path_distances(self, use_start_point=False, neurite_type=TreeType.all):
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
        return self._neurite_loop(lambda t: mtr.i_section_path_length(t, use_start_point),
                                  neurite_type=neurite_type)

    def section_radial_distances(self, origin=None, use_start_point=False,
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
        return self._neurite_loop(lambda t: mtr.i_section_radial_dist(t, origin,
                                                                      use_start_point),
                                  neurite_type=neurite_type)

    def trunk_origin_radii(self, neurite_type=TreeType.all):
        '''Get the trunk origin radii of a given type in a neuron'''
        return self._iterable_type(
            [mtr.trunk_origin_radius(t.tree) for t in self._neurites
             if checkTreeType(neurite_type, t.type)]
        )

    def trunk_section_lengths(self, neurite_type=TreeType.all):
        '''Get the trunk section lengths of a given type in a neuron'''
        return self._iterable_type(
            [mtr.trunk_section_length(t.tree) for t in self._neurites
             if checkTreeType(neurite_type, t.type)]
        )

    def n_neurites(self, neurite_type=TreeType.all):
        '''Get the number of neurites of a given type in a neuron'''
        return sum(1 for n in self._neurites
                   if checkTreeType(neurite_type, n.type))


class NeuronFeatures(NeuriteFeatures):
    ''' NeuronFeatures Class
    '''
    def __init__(self, somata, neurites, iterable_type=np.array):
        ''' Construct a NeuronFeatures Object
        '''
        super(NeuronFeatures, self).__init__(neurites, iterable_type=iterable_type)
        self._somata = somata

    def soma_radius(self):
        '''Get the radius of the soma'''
        return self._pkg(soma.radius for soma in self._somata)

    def soma_surface(self):
        '''Get the surface area of the soma.

        Note:
            The surface area is calculated by assuming the soma is spherical.
        '''
        return self._pkg(4. * np.pi * soma.radius ** 2 for soma in self._somata)


class PopulationFeatures(NeuronFeatures):
    ''' PopulationFeatures Class
    '''
    def __init__(self, somata, neurites, iterable_type=np.array):
        ''' Construct a PopulationFeatures object
        '''
        super(PopulationFeatures, self).__init__(somata, neurites, iterable_type=iterable_type)


class Neurite(object):
    ''' Neurite Class
    '''
    def __init__(self, tree):
        ''' Construct Neurite object
        '''
        self._tree = tree
        self._type = mtr.get_tree_type(tree)
        self.features = NeuriteFeatures([self, ])

    @property
    def tree(self):
        ''' Returns stored tree object
        '''
        return self._tree

    @property
    def type(self):
        ''' Returns neurite type
        '''
        return self._type

    @property
    def copy(self):
        ''' Returns a copy of the neurite
        '''
        pass


class Neuron(object):
    ''' Neuron Class
    '''
    def __init__(self, soma, neurites, name='Neuron'):
        '''Construct a Neuron

        Arguments:
            soma: soma object
            neurites: iterable of neurite tree structures.
            name: Optional name for this Neuron.
        '''
        self.soma = soma
        self.neurites = neurites
        self.features = NeuronFeatures(soma, neurites)
        self.name = name

    @property
    def bounding_box(self):
        '''Return 3D bounding box of a neuron

        Returns:
            2D numpy array of [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        '''

        # Get the bounding coordinates of the neurites
        nmin_xyz, nmax_xyz = (np.array([np.inf, np.inf, np.inf]),
                              np.array([np.NINF, np.NINF, np.NINF]))

        for p in tr.i_chain(self.neurites, tr.ipreorder, lambda p: p):
            nmin_xyz = np.minimum(p[:COLS.R], nmin_xyz)
            nmax_xyz = np.maximum(p[:COLS.R], nmax_xyz)

        # Get the bounding coordinates of the soma
        smin_xyz = np.array(self.soma.center) - self.soma.radius

        smax_xyz = np.array(self.soma.center) + self.soma.radius

        return np.array([np.minimum(smin_xyz, nmin_xyz),
                         np.maximum(smax_xyz, nmax_xyz)])

    @property
    def copy(self):
        '''Return a copy of the Neuron object.
        '''
        pass


class Population(object):
    '''Neuron Population Class'''
    def __init__(self, neurons, name='Population'):
        '''Construct a Population

        Arguments:
            neurons: iterable of neuron objects (core or ezy) .
            name: Optional name for this Population.
        '''
        self.neurons = neurons
        self.somata = [neu.soma for neu in neurons]
        self.neurites = list(chain(*(neu.neurites for neu in neurons)))
        self.features = PopulationFeatures(self.somata, self.neurites)
        self.name = name

    @property
    def copy(self):
        ''' Returns a copy fo the population
        '''
        pass

if __name__ == '__main__':
    from neurom.core.tree import Tree
    from neurom.core.neuron import make_soma

    TREE = Tree([0.0, 0.0, 0.0, 1.0, 1, 1, 2])
    T1 = TREE.add_child(Tree([0.0, 1.0, 0.0, 1.0, 1, 1, 2]))
    T2 = T1.add_child(Tree([0.0, 2.0, 0.0, 1.0, 1, 1, 2]))
    T3 = T2.add_child(Tree([0.0, 4.0, 0.0, 2.0, 1, 1, 2]))
    T4 = T3.add_child(Tree([0.0, 5.0, 0.0, 2.0, 1, 1, 2]))
    T5 = T4.add_child(Tree([2.0, 5.0, 0.0, 1.0, 1, 1, 2]))
    T6 = T4.add_child(Tree([0.0, 5.0, 2.0, 1.0, 1, 1, 2]))
    T7 = T5.add_child(Tree([3.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T8 = T7.add_child(Tree([4.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T9 = T6.add_child(Tree([0.0, 5.0, 3.0, 0.75, 1, 1, 2]))
    T10 = T9.add_child(Tree([0.0, 6.0, 3.0, 0.75, 1, 1, 2]))

    sm = make_soma([[0, 0, 0, 1, 1, 1, -1]])
    nrts = [Neurite(tr) for tr in [TREE, TREE, TREE]]
    nrn = Neuron(sm, nrts)
    pop = Population([nrn, nrn, nrn, nrn])

    print pop.features.section_radial_distances()
    print
    print pop.neurons[0].features.section_radial_distances()
    print
    print pop.neurons[0].neurites[0].features.section_radial_distances()
