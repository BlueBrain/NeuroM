# Copyright (c) 2015, Ecole Polytechnique Federal de Lausanne, Blue Brain Project
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
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''Basic functions used for tree analysis'''
from neurom.core import tree as tr
from neurom.analysis.morphmath import point_dist
from neurom.analysis.morphmath import path_distance
from neurom.core.dataformat import COLS
from neurom.core.tree import val_iter
from neurom.core.tree import iter_preorder
import numpy as np


def segment_length(seg):
    '''Return the length of a segment.

    Returns: Euclidian distance between centres of points in seg
    '''
    return point_dist(seg[0], seg[1])


def segment_diameter(seg):
    '''Return the mean diameter of a segment

    Returns: arithmetic mean of the diameters of the points in seg
    '''
    return seg[0][COLS.R] + seg[1][COLS.R]


def segment_radial_dist(seg, pos):
    '''Return the radial distance of a tree segment to a given point

    The radial distance is the euclidian distance between the mid-point of
    the segment and the point in question.

    Args:
        seg: tree segment

        pos: origin to which disrances are measured. It must have at lease 3
        components. The first 3 components are (x, y, z).
    '''
    return point_dist(pos, np.divide(np.add(seg[0], seg[1]), 2.0))


def path_length(tree):
    '''Get the path length from a sub-tree to the root node'''
    return np.sum(point_dist(s[0], s[1])
                  for s in tr.iter_segment(tree, tr.iter_upstream))


def get_segment_lengths(tree):
    ''' return a list of segments length inside tree
    '''
    return [segment_length(s) for s in tr.iter_segment(tree)]


def get_segment_diameters(tree):
    ''' return a list of segments diameter inside tree
    '''
    return [segment_diameter(s) for s in tr.iter_segment(tree)]


def get_segment_radial_dists(pos, tree):
    '''Return a list of radial distances of tree segments to a given point

    Thr radial distance is the euclidian distance between the mid-point of
    the segment and the point in question.

    Args:
        pos: origin to which disrances are measured. It must have at lease 3
        components. The first 3 components are (x, y, z).

        tree: tree of raw data rows.

    '''
    return [segment_radial_dist(s, pos)
            for s in tr.iter_segment(tree)]


def find_tree_type(tree):

    """
    Calculates the 'mean' type of the tree.
    Accepted tree types are:
    'undefined', 'axon', 'basal', 'apical'
    The 'mean' tree type is defined as the type
    that is shared between at least 51% of the tree's points.
    Returns a tree with a tree.type defined.
    """

    tree_types = ['undefined', 'soma', 'axon', 'basal', 'apical']

    types = [node.value[COLS.TYPE] for node in tr.iter_preorder(tree)]

    tree.type = tree_types[int(np.median(types))]


def get_tree_type(tree):

    """
    If tree does not have a tree type,
    calculates the tree type and returns it.
    If tree has a tree type returns its precomputed type.
    """

    if not hasattr(tree, 'type'):
        find_tree_type(tree)

    return tree.type


def get_section_lengths(tree):
    """
    Compute the length of section on this tree
    """
    return [path_distance([p.value for p in sec]) for sec in tr.iter_section(tree)]


def get_section_number(tree):
    """
    Return number of section on tree
    """
    return len(list(tr.iter_section(tree)))


def get_bounding_box(tree):
    """
    Returns the boundaries of the tree
    in three dimensions:
    [[xmin, ymin, zmin],
    [xmax, ymax, zmax]]
    """

    min_x = min(p[0] for p in val_iter(iter_preorder(tree)))
    min_y = min(p[1] for p in val_iter(iter_preorder(tree)))
    min_z = min(p[2] for p in val_iter(iter_preorder(tree)))
    max_x = max(p[0] for p in val_iter(iter_preorder(tree)))
    max_y = max(p[1] for p in val_iter(iter_preorder(tree)))
    max_z = max(p[2] for p in val_iter(iter_preorder(tree)))

    return np.array([[min_x, min_y, min_z], [max_x, max_y, max_z]])
