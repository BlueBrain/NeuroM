'''Basic functions used for tree analysis'''
from neurom.core import tree as tr
from neurom.analysis.morphmath import point_dist
from neurom.core.point import point_from_row
from neurom.core.point import Point
from neurom.core.dataformat import COLS
import numpy as np


def get_segment_lengths(tree):
    ''' return a list of segments length inside tree
    '''
    return [point_dist(point_from_row(s[0]), point_from_row(s[1])) for s in tr.iter_segment(tree)]


def get_segment_diameters(tree):
    ''' return a list of segments diameter inside tree
    '''
    return [point_from_row(s[0]).r + point_from_row(s[1]).r for s in tr.iter_segment(tree)]


def get_segment_radialdists(position, tree):
    ''' return a list of radial distance of segment to given point'''
    pos = point_from_row(position)
    return [point_dist(pos, Point((point_from_row(s[0]).x + point_from_row(s[1]).x) / 2.0,
                                  (point_from_row(s[0]).y + point_from_row(s[1]).y) / 2.0,
                                  (point_from_row(s[0]).z + point_from_row(s[1]).z) / 2.0,
                                  0.0, 0.0)) for s in tr.iter_segment(tree)]


def find_tree_type(tree):

    """
    Calculates the 'mean' type of the tree.
    Accepted tree types are:
    'undefined', 'axon', 'basal', 'apical'
    The 'mean' tree type is defined as the type
    that is shared between at least 51% of the tree's points.
    Returns a tree with a tree.type defined.
    """

    types = []

    tree_types = ['undefined', 'soma', 'axon', 'basal', 'apical']

    for raw_point in tr.iter_preorder(tree):

        types.append(raw_point.value[COLS.TYPE])

    tree.type = tree_types[int(np.median(types))]

    return tree


def get_tree_type(tree):

    """
    If tree does not have a tree type,
    calculates the tree type and returns it.
    If tree has a tree type returns its precomputed type.
    """

    if not hasattr(tree, 'type'):
        tree = find_tree_type(tree)

    return tree.type
