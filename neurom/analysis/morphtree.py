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

'''Basic functions used for tree analysis'''
from itertools import imap
from neurom.core import tree as tr
from neurom.core.types import TreeType
from neurom.analysis.morphmath import point_dist
from neurom.analysis.morphmath import path_distance
from neurom.analysis.morphmath import angle_3points
from neurom.core.dataformat import COLS
from neurom.core.tree import val_iter
from neurom.core.tree import ipreorder
from neurom.core.tree import i_branch_end_points
import numpy as np
import logging

LOG = logging.getLogger(__name__)


def segment_length(seg):
    '''Return the length of a segment.

    Returns: Euclidian distance between centres of points in seg
    '''
    return point_dist(seg[0], seg[1])


def segment_radius(seg):
    '''Return the mean radius of a segment

    Returns: arithmetic mean of the radii of the points in seg
    '''
    return (seg[0][COLS.R] + seg[1][COLS.R]) / 2.


def segment_radial_dist(seg, pos):
    '''Return the radial distance of a tree segment to a given point

    The radial distance is the euclidian distance between the mid-point of
    the segment and the point in question.

    Parameters:
        seg: tree segment

        pos: origin to which disrances are measured. It must have at lease 3
        components. The first 3 components are (x, y, z).
    '''
    return point_dist(pos, np.divide(np.add(seg[0], seg[1]), 2.0))


def path_length(tree):
    '''Get the path length from a sub-tree to the root node'''
    return np.sum(segment_length(s)
                  for s in tr.val_iter(tr.isegment(tree, tr.iupstream)))


def i_segment_length(tree):
    ''' return an iterator of tree segment lengths
    '''
    return imap(segment_length, tr.val_iter(tr.isegment(tree)))


def i_segment_radius(tree):
    ''' return an iterator of tree segment radii
    '''
    return imap(segment_radius, tr.val_iter(tr.isegment(tree)))


def i_segment_radial_dist(pos, tree):
    '''Return an iterator of radial distances of tree segments to a given point

    The radial distance is the euclidian distance between the mid-point of
    the segment and the point in question.

    Parameters:
        pos: origin to which disrances are measured. It must have at least 3
        components. The first 3 components are (x, y, z).

        tree: tree of raw data rows.

    '''
    return imap(lambda s: segment_radial_dist(s, pos),
                tr.val_iter(tr.isegment(tree)))


def i_segment_meander_angle(tree):
    '''Return an iterator to a tree meander angle

    The meander angle is defined as the angle between to adjacent  segments.
    Applies neurom.morphmath.angle_3points to triplets of
    '''

    return imap(lambda t: angle_3points(t[1], t[0], t[2]),
                tr.val_iter(tr.itriplet(tree)))


def i_local_bifurcation_angle(tree):
    '''Return the opening angle between two out-going segments
    in a bifurcation point
    '''
    return imap(lambda t: angle_3points(t.value,
                                        t.children[0].value,
                                        t.children[1].value),
                tr.ibifurcation_point(tree))


def i_remote_bifurcation_angle(tree):
    '''Return the opening angle between the last segments of two out-going
    sections of a bifurcation point
    '''
    def _remangle(t):
        '''Helper to calculate the remote angle'''
        end_points = tuple(p for p in i_branch_end_points(t))
        return angle_3points(t.value, end_points[0].value, end_points[1].value)

    return imap(_remangle, tr.ibifurcation_point(tree))


def i_section_radial_dist(tree, pos=None, use_start_point=False):
    '''Return an iterator of radial distances of tree sections to a given point

    The radial distance is the euclidian distance between the either the
    end-point or rhe start point of the section and the point in question.

    Parameters:
        tree: tree object
        pos: origin to which distances are measured. It must have at least 3\
            components. The first 3 components are (x, y, z).\
            (default tree origin)

        use_start_point: If true, calculate distance from section start point,\
            else from end-point (default, False)

    '''
    pos = tree.value if pos is None else pos
    sec_idx = 0 if use_start_point else -1
    return imap(lambda s: point_dist(s[sec_idx], pos),
                tr.val_iter(tr.isection(tree)))


def i_section_path_length(tree, use_start_point=False):
    '''Return an iterator of path lengths of tree sections

    Path lengths are measured to the tree's root.

    Parameters:
        tree: tree object
        use_start_point: If true, calculate path length from section start point,\
            else from end-point (default, False)

    '''
    sec_idx = 0 if use_start_point else -1
    return imap(lambda s: path_length(s[sec_idx]), tr.isection(tree))


def find_tree_type(tree):

    """
    Calculates the 'mean' type of the tree.
    Accepted tree types are:
    'undefined', 'axon', 'basal', 'apical'
    The 'mean' tree type is defined as the type
    that is shared between at least 51% of the tree's points.
    Returns:
        The type of the tree
    """

    tree_types = tuple(TreeType)

    types = [node[COLS.TYPE] for node in tr.val_iter(tr.ipreorder(tree))]

    return tree_types[int(np.median(types))]


def set_tree_type(tree):
    ''' Set the type of a tree as 'type' attribute'''
    tree.type = find_tree_type(tree)


def get_tree_type(tree):
    '''Return a tree's type

    If tree does not have a type, calculate it.
    Otherwise, use its pre-computed value.
    '''

    if not hasattr(tree, 'type'):
        set_tree_type(tree)

    return tree.type


def i_section_length(tree):
    """
    Return an iterator of tree section lengths
    """
    return imap(path_distance, tr.val_iter(tr.isection(tree)))


def n_sections(tree):
    """
    Return number of sections in tree
    """
    return sum(1 for _ in tr.isection(tree))


def get_bounding_box(tree):
    """
    Returns the boundaries of the tree
    in three dimensions:
    [[xmin, ymin, zmin],
    [xmax, ymax, zmax]]
    """

    min_x = min(p[0] for p in val_iter(ipreorder(tree)))
    min_y = min(p[1] for p in val_iter(ipreorder(tree)))
    min_z = min(p[2] for p in val_iter(ipreorder(tree)))
    max_x = max(p[0] for p in val_iter(ipreorder(tree)))
    max_y = max(p[1] for p in val_iter(ipreorder(tree)))
    max_z = max(p[2] for p in val_iter(ipreorder(tree)))

    return np.array([[min_x, min_y, min_z], [max_x, max_y, max_z]])
