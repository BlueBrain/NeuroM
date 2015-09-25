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

'''Basic functions used for tree analysis

These functions all depend on the internal structure of the tree or its
different iteration modes.
'''
from itertools import imap
from neurom.core import tree as tr
from neurom.core.types import TreeType
import neurom.analysis.morphmath as mm
from neurom.core.dataformat import COLS
from neurom.core.tree import val_iter
import numpy as np
import logging

LOG = logging.getLogger(__name__)


def path_length(tree):
    '''Get the path length from a sub-tree to the root node'''
    return np.sum(s for s in
                  tr.imap_val(mm.segment_length, tr.isegment(tree, tr.iupstream)))


def local_bifurcation_angle(bifurcation_point):
    '''Return the opening angle between two out-going segments
    in a bifurcation point
    '''
    return mm.angle_3points(bifurcation_point.value,
                            bifurcation_point.children[0].value,
                            bifurcation_point.children[1].value)


def branch_order(tree_section):
    '''Branching order of a tree section

    The branching order is defined as the depth of the tree section.

    Note:
        The first level has branch order 0.
    '''
    node = tree_section[-1]
    bo = sum(1 for _ in tr.iforking_point(node, tr.iupstream))
    return bo - 2 if tr.is_forking_point(node) else bo - 1


def i_segment_length(tree):
    ''' return an iterator of tree segment lengths
    '''
    return tr.imap_val(mm.segment_length, tr.isegment(tree))


def i_segment_radius(tree):
    ''' return an iterator of tree segment radii
    '''
    return tr.imap_val(mm.segment_radius, tr.isegment(tree))


def i_segment_volume(tree):
    ''' return an iterator of tree segment volumes
    '''
    return tr.imap_val(mm.segment_volume, tr.isegment(tree))


def i_segment_area(tree):
    ''' return an iterator of tree segment areas
    '''
    return tr.imap_val(mm.segment_area, tr.isegment(tree))


def i_segment_radial_dist(pos, tree):
    '''Return an iterator of radial distances of tree segments to a given point

    The radial distance is the euclidian distance between the mid-point of
    the segment and the point in question.

    Parameters:
        pos: origin to which disrances are measured. It must have at least 3
        components. The first 3 components are (x, y, z).

        tree: tree of raw data rows.

    '''
    return tr.imap_val(lambda s: mm.segment_radial_dist(s, pos), tr.isegment(tree))


def i_segment_meander_angle(tree):
    '''Return an iterator to a tree meander angle

    The meander angle is defined as the angle between to adjacent  segments.
    Applies neurom.morphmath.angle_3points to triplets of
    '''

    return tr.imap_val(lambda t: mm.angle_3points(t[1], t[0], t[2]), tr.itriplet(tree))


def i_local_bifurcation_angle(tree):
    '''Return the opening angle between two out-going segments
    in a bifurcation point
    '''
    return imap(local_bifurcation_angle,
                tr.ibifurcation_point(tree))


def i_remote_bifurcation_angle(tree):
    '''Return the opening angle between the last segments of two out-going
    sections of a bifurcation point
    '''
    def _remangle(t):
        '''Helper to calculate the remote angle'''
        end_points = tuple(p for p in tr.i_branch_end_points(t))
        return mm.angle_3points(t.value, end_points[0].value, end_points[1].value)

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
    return tr.imap_val(lambda s: mm.point_dist(s[sec_idx], pos), tr.isection(tree))


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
    return tr.imap_val(mm.section_length, tr.isection(tree))


def n_sections(tree):
    """
    Return number of sections in tree
    """
    return sum(1 for _ in tr.isection(tree))


def get_bounding_box(tree):
    """
    Returns:
        The boundaries of the tree in three dimensions:
            [[xmin, ymin, zmin],
            [xmax, ymax, zmax]]
    """

    min_xyz, max_xyz = (np.array([np.inf, np.inf, np.inf]),
                        np.array([np.NINF, np.NINF, np.NINF]))

    for p in val_iter(tr.ipreorder(tree)):
        min_xyz = np.minimum(p[:COLS.R], min_xyz)
        max_xyz = np.maximum(p[:COLS.R], max_xyz)

    return np.array([min_xyz, max_xyz])
