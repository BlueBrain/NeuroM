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

'''Dendrogram helper functions and class'''

from neurom.core import Tree
from neurom.point_neurite.core import Neuron
from neurom.morphmath import segment_length
from neurom.core.dataformat import COLS
from .treefunc import n_terminations

import numpy as np
import sys


def _max_recursion_depth(obj):
    ''' Estimate recursion depth, which is defined as the number of nodes in a tree
    '''
    neurites = obj.neurites if hasattr(obj, 'neurites') else [obj]

    return max(sum(1 for _ in neu.ipreorder()) for neu in neurites)


def _total_rectangles(tree):
    '''
    Calculate the total number of segments that are required
    for the dendrogram. There is a vertical line for each segment
    and two horizontal line at each branching point
    '''
    def f(children):
        '''Calculates number of lines needed for the children of a node
        '''
        return 2 * len(children) if len(children) != 1 else 1

    return sum(f(node.children) for node in tree.ipreorder())


def _n_rectangles(obj):
    '''
    Calculate the total number of rectangles with respect to
    the type of the object
    '''
    if isinstance(obj, Tree):

        return _total_rectangles(obj)

    elif isinstance(obj, Neuron):

        # + 1 accounts for the rectangle needed to represent the soma
        return sum(_total_rectangles(neu) for neu in obj.neurites) + 1

    else:

        return 0


def _square_segment(radius, origin):
    '''Vertices for a square
    '''
    return np.array(((origin[0] - radius, origin[1] - radius),
                     (origin[0] - radius, origin[1] + radius),
                     (origin[0] + radius, origin[1] + radius),
                     (origin[0] + radius, origin[1] - radius)))


def _vertical_segment(old_offs, new_offs, spacing, radii):
    '''Vertices for a vertical rectangle
    '''
    return np.array(((new_offs[0] - radii[0], old_offs[1] + spacing[1]),
                     (new_offs[0] - radii[1], new_offs[1]),
                     (new_offs[0] + radii[1], new_offs[1]),
                     (new_offs[0] + radii[0], old_offs[1] + spacing[1])))


def _horizontal_segment(old_offs, new_offs, spacing, diameter):
    '''Vertices of a horizontal rectangle
    '''
    return np.array(((old_offs[0], old_offs[1] + spacing[1]),
                     (new_offs[0], old_offs[1] + spacing[1]),
                     (new_offs[0], old_offs[1] + spacing[1] - diameter),
                     (old_offs[0], old_offs[1] + spacing[1] - diameter)))


def _spacingx(node, max_dims, xoffset, xspace):
    '''Determine the spacing of the current node depending on the number
       of the leaves of the tree
    '''
    x_spacing = n_terminations(node) * xspace

    if x_spacing > max_dims[0]:
        max_dims[0] = x_spacing

    return xoffset - x_spacing / 2.


def _update_offsets(start_x, spacing, terminations, offsets, length):
    '''Update the offsets
    '''
    return (start_x + spacing[0] * terminations / 2.,
            offsets[1] + spacing[1] * 2. + length)


def _max_diameter(tree):
    '''Find max diameter in tree
    '''
    return 2. * max(node.value[COLS.R] for node in tree.ipreorder())


class Dendrogram(object):
    '''Dendrogram
    '''

    def __init__(self, obj, show_diameters=True):
        '''Create dendrogram
        '''

        # flag for diameters
        self._show_diameters = show_diameters

        # input object, tree, or neuron
        self._obj = obj

        # counter/index for the storage of the rectangles.
        # it is updated recursively
        self._n = 0

        # the maximum lengths in x and y that is occupied
        # by a neurite. It is updated recursively.
        self._max_dims = [0., 0.]

        # stores indices that refer to the _rectangles array
        # for each neurite
        self._groups = []

        # dims store the max dimensions for each neurite
        # essential for the displacement in the plotting
        self._dims = []

        # initialize the number of rectangles
        self._rectangles = np.zeros([_n_rectangles(self._obj), 4, 2])

        # determine the maximum recursion depth for the given object
        # which depends on the tree with the maximum number of nodes
        self._max_rec_depth = _max_recursion_depth(self._obj)

    def _generate_soma(self):
        '''soma'''
        radius = self._obj.soma.radius
        return _square_segment(radius, (0., -radius))

    def generate(self):
        '''Generate dendrogram
        '''
        offsets = (0., 0.)

        n_previous = 0

        # set recursion limit with respect to
        # the max number of nodes on the trees
        old_depth = sys.getrecursionlimit()
        max_depth = old_depth if old_depth > self._max_rec_depth else self._max_rec_depth
        sys.setrecursionlimit(max_depth)

        if isinstance(self._obj, Tree):

            max_diameter = _max_diameter(self._obj)

            self._generate_dendro(self._obj, (max_diameter, 0.), offsets)

            self._groups.append((0., self._n))

            self._dims.append(self._max_dims)

        else:

            for neurite in self._obj.neurites:

                max_diameter = _max_diameter(neurite)

                self._generate_dendro(neurite, (max_diameter, 0.), offsets)

                # store in trees the indices for the slice which corresponds
                # to the current neurite
                self._groups.append((n_previous, self._n))

                # store the max dims per neurite for view positioning
                self._dims.append(self._max_dims)

                # reset the max dimensions for the next tree in line
                self._max_dims = [0., 0.]

                # keep track of the next tree start index in list
                n_previous = self._n

        # set it back to its initial value
        sys.setrecursionlimit(old_depth)

    def _generate_dendro(self, current_node, spacing, offsets):
        '''Recursive function for dendrogram line computations
        '''
        max_dims = self._max_dims
        start_x = _spacingx(current_node, max_dims, offsets[0], spacing[0])

        radii = [0., 0.]
        # store the parent radius in order to construct polygonal segments
        # isntead of simple line segments
        radii[0] = current_node.value[COLS.R] if self._show_diameters else 0.

        for child in current_node.children:

            # segment length
            ln = segment_length((current_node.value, child.value))

            # extract the radius of the child node. Need both radius for
            # realistic segment representation
            radii[1] = child.value[COLS.R] if self._show_diameters else 0.

            # number of leaves in child
            terminations = n_terminations(child)

            # horizontal spacing with respect to the number of
            # terminations
            new_offsets = _update_offsets(start_x, spacing, terminations, offsets, ln)

            # create and store vertical segment
            self._rectangles[self._n] = _vertical_segment(offsets, new_offsets, spacing, radii)

            # assign segment id to color array
            # colors[n[0]] = child.value[4]
            self._n += 1

            if offsets[1] + spacing[1] * 2 + ln > max_dims[1]:
                max_dims[1] = offsets[1] + spacing[1] * 2. + ln

            self._max_dims = max_dims
            # recursive call to self.
            self._generate_dendro(child, spacing, new_offsets)

            # update the starting position for the next child
            start_x += terminations * spacing[0]

            # write the horizontal lines only for bifurcations, where the are actual horizontal
            # lines and not zero ones
            if offsets[0] != new_offsets[0]:

                # horizontal segment. Thickness is either 0 if show_diameters is false
                # or 1. if show_diameters is true
                self._rectangles[self._n] = _horizontal_segment(offsets, new_offsets, spacing, 0.)
                self._n += 1

    @property
    def data(self):
        ''' Returns the array with the rectangle collection
        '''
        return self._rectangles

    @property
    def groups(self):
        ''' Returns the list of the indices for the slicing of the
            rectangle array wich correspond to each neurite
        '''
        return self._groups

    @property
    def dims(self):
        ''' Returns the list of the max dimensions for each neurite
        '''
        return self._dims

    @property
    def types(self):
        ''' Returns an iterator over the types of the neurites in the object.
            If the object is a tree, then one value is returned.
        '''
        neurites = self._obj.neurites if hasattr(self._obj, 'neurites') else (self._obj,)
        return (neu.type for neu in neurites)

    @property
    def soma(self):
        ''' Returns soma
        '''
        return self._generate_soma() if hasattr(self._obj, 'soma') else None
