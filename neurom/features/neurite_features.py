from itertools import product
from neurom.core.types import TreeType
from neurom.core.types import tree_type_checker as _ttc
from neurom import segments as _seg
from neurom import sections as _sec
from neurom import bifurcations as _bifs
from neurom import points as _pts
from neurom.core.neuron import Neuron as CoreNeuron
from neurom.analysis.morphtree import i_section_radial_dist
from neurom.analysis.morphtree import trunk_section_length
from neurom.analysis.morphtree import compare_trees
import math
import numpy as np
from neurom import iter_neurites
from functools import partial


def section_lengths(neurites, neurite_type=TreeType.all):
    '''Get an iterable containing the lengths of all sections of a given type'''
    return iter_neurites(neurites, _sec.length, _ttc(neurite_type))


def section_number(neurites, neurite_type=TreeType.all):
    '''Get the number of sections of a given type'''
    yield _sec.count(neurites, _ttc(neurite_type))


def per_neurite_section_number(obj, neurite_type=TreeType.all):
        '''Get an iterable with the number of sections for a given neurite type'''
        neurites = ([obj] if isinstance(obj, TreeType)
                else (obj.neurites if hasattr(obj, 'neurites') else obj))
        return (_sec.count(n) for n in neurites if _ttc(neurite_type)(n))


def section_path_distances(neurites, use_start_point=False,
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
    magic_iter = (_sec.start_point_path_length if use_start_point
                  else _sec.end_point_path_length)
    return iter_neurites(neurites, magic_iter, _ttc(neurite_type))


def segment_lengths(neurites, neurite_type=TreeType.all):
    '''Get an iterable containing the lengths of all segments of a given type'''
    return iter_neurites(neurites, _sec.length, _ttc(neurite_type))


def local_bifurcation_angles(neurites, neurite_type=TreeType.all):
    '''Get local bifircation angles of all segments of a given type

    The local bifurcation angle is defined as the angle between
    the first segments of the bifurcation branches.

    Returns:
        Iterable containing bifurcation angles in radians
    '''
    return iter_neurites(neurites, _bifs.local_angle, _ttc(neurite_type))


def remote_bifurcation_angles(neurites, neurite_type=TreeType.all):
    '''Get remote bifircation angles of all segments of a given type

    The remote bifurcation angle is defined as the angle between
    the lines joining the bifurcation point to the last points
    of the bifurcated sections.

    Returns:
        Iterable containing bifurcation angles in radians
    '''
    return iter_neurites(neurites, _bifs.remote_angle, _ttc(neurite_type))


def neurite_number(obj, neurite_type=TreeType.all):
    '''Get the number of neurites of a given type in a neuron'''
    neurites = ([obj] if isinstance(obj, TreeType)
                else (obj.neurites if hasattr(obj, 'neurites') else obj))
    yield sum(1. for n in neurites if _ttc(neurite_type)(n))
