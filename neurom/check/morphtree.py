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

'''
Python module of NeuroM to check neuronal trees.
'''

import numpy as np
from neurom.core.tree import ipreorder, isection
from neurom.core.dataformat import COLS
from neurom.analysis import morphmath as mm
from neurom.analysis.morphtree import principal_direction_extent
from neurom.core.point import COLS


def is_monotonic(tree, tol):
    '''Check if tree is monotonic, i.e. if each child has smaller or
        equal diameters from its parent

        Arguments:
            tree : tree object
            tol: numerical precision
    '''

    for node in ipreorder(tree):

        if node.parent is not None:

            if node.value[COLS.R] > node.parent.value[COLS.R] + tol:

                return False

    return True


def is_flat(tree, tol, method='tolerance'):
    '''Check if neurite is flat using the given method

        Input

            tree : the tree object

            tol : tolerance

            method : the method of flatness estimation.
            'tolerance' returns true if any extent of the tree
            is smaller than the given tolerance
            'ratio' returns true if the ratio of the smallest directions
            is smaller than tol. e.g. [1,2,3] -> 1/2 < tol

        Returns

            True if it is flat

    '''

    ext = principal_direction_extent(tree)

    if method == 'ratio':

        sorted_ext = np.sort(ext)
        return sorted_ext[0] / sorted_ext[1] < float(tol)

    else:

        return any(ext < float(tol))


def is_back_tracking(tree):
    ''' zigzag
    '''
    def paired(n):
        ''' Pairs the input list into triplets
        '''
        return zip(n, n[1:], n[2:])

    def coords(node):
        ''' Returns the first three values of the tree that
        correspond to the x, y, z coordinates
        '''
        return node.value[:COLS.R]

    def max_radius(seg):
        ''' Returns maximum radius from the two segment endpoints
        '''
        return max(seg[0].value[COLS.R], seg[1].value[COLS.R])

    def is_not_zero_seg(seg):
        ''' Returns True if segment has zero length
        '''
        return not np.allclose(coords(seg[0]), coords(seg[1]))

    def is_in_the_same_verse(seg, other):
        ''' Checks if the vectors face the same direction. This
        is true if their dot product is greater than zero.
        '''
        return np.dot(coords(other[1]) - coords(other[0]),
                      coords(seg[1]) - coords(seg[0])) > 0.

    def is_seg_within_seg_radius(dist, seg, other):
        ''' Checks whether the orthogonal distance from the point at the end of
        a segment to the other's segment body is smaller than the sum of their radii
        '''
        return dist <= max_radius(seg) + max_radius(other)

    def is_seg_projection_within_other(seg, other):
        '''Checks if a segment is in proximity of another one upstream
        '''
        s1 = coords(other[0])
        s2 = coords(other[1])

        # center of the other segment
        C = 0.5 * (s1 + s2)

        # endpoint of the current segment
        P = coords(seg[1])

        # vector from C to P
        CP = P - C

        # projection of CP upon the other segment vector
        prj = mm.vector_projection(CP, s2 - s1)

        # check if the distance of the orthogonal complement of CP projection on S1S2
        # is smaller than the sum of the radii. If not exit early.
        if not is_seg_within_seg_radius(np.linalg.norm(CP - prj), seg, other):
            return False

        length = np.linalg.norm(s2 - s1)
        # projection lies within the length of the cylinder
        # check if the distance between the center of other segment
        # and the projection of the end point of the current one
        # is smaller than half of the others length plus a 5% tolerance
        return np.linalg.norm(prj) < 0.55 * length

    def is_inside_cylinder(seg, other):
        ''' Checks if a point approximately lies within a cylindrical
        volume
        '''
        return not is_in_the_same_verse(seg, other) and \
               is_seg_projection_within_other(seg, other)

    # filter out single segment sections
    for segments in (paired(nodes) for nodes in isection(tree) if len(nodes) > 2):
        # filter out zero length segments
        for i, seg in enumerate(filter(is_not_zero_seg, segments[1:])):
            # check if the end point of the segment lies within the previous
            # ones in the current section
            if any(is_inside_cylinder(seg, other) for other in segments[:i + 1]):
                return True

    return False
