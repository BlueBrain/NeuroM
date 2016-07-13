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

'''Neurite functions'''

from functools import wraps, partial
from itertools import imap, chain, izip
import numpy as np
from neurom.core.tree import (ipreorder,
                              ibifurcation_point,
                              iforking_point,
                              iupstream,
                              ileaf)
from neurom.core.types import tree_type_checker as is_type
from neurom.core.types import NeuriteType
from neurom.analysis import morphmath as mm
from . import Neurite
from .sectionfunc import branch_order, section_radial_distance
from ._bifurcationfunc import (local_bifurcation_angle,
                               remote_bifurcation_angle,
                               bifurcation_partition)


def neurite_fun(fun):
    '''Wrapper to extract the neurite trees from the first parameter'''
    @wraps(fun)
    def _neurites(obj, **kwargs):
        '''Extract neurites from obj and forward to wrapped function'''
        if hasattr(obj, 'neurites'):
            obj = obj.neurites
        elif isinstance(obj, Neurite):
            obj = (obj,)
        return fun(obj, **kwargs)

    return _neurites


@neurite_fun
def iter_sections(neurites, iterator_type=ipreorder, neurite_filter=None):
    '''Returns an iterator to the nodes in a iterable of neurite objects

    Parameters:
        neurites: neuron, population, neurite, or iterable containing neurite objects
        iterator_type: type of the iteration (ipreorder, iupstream, ibifurcation_point)
        neurite_filter: optional top level filter on properties of neurite neurite objects.
    '''
    neurites = (neurites if neurite_filter is None
                else filter(neurite_filter, neurites))

    trees = (t.root_node if isinstance(t, Neurite) else t for t in neurites)

    return chain.from_iterable(imap(iterator_type, trees))


def iter_segments(neurites, neurite_filter=None):
    '''Return an iterator to the segments in a collection of neurites

    Parameters:
        neurites: neuron, population, neurite, or iterable containing neurite objects
        neurite_filter: optional top level filter on properties of neurite neurite objects.

    Note:
        This is a convenience function provideded for generic access to
        neuron segments. It may have a performance overhead WRT custom
        made segment analysis functions that leverage numpy.
    '''
    return chain(s for ss in iter_sections(neurites, neurite_filter=neurite_filter)
                 for s in izip(ss.points[:-1], ss.points[1:]))


def n_segments(neurites, neurite_type=NeuriteType.all):
    '''Number of segments in a collection of neurites'''
    return sum(len(s.points) - 1
               for s in iter_sections(neurites, neurite_filter=is_type(neurite_type)))


@neurite_fun
def n_neurites(neurites, neurite_type=NeuriteType.all):
    '''Number of neurites in a collection of neurites'''
    is_ntype = is_type(neurite_type)
    return sum(1 for n in neurites if is_ntype(n))


@neurite_fun
def n_sections(neurites, iterator_type=ipreorder, neurite_type=NeuriteType.all):
    '''Number of sections in a collection of neurites'''
    return sum(1 for _ in iter_sections(neurites,
                                        iterator_type=iterator_type,
                                        neurite_filter=is_type(neurite_type)))


n_bifurcation_points = partial(n_sections, iterator_type=ibifurcation_point)
n_forking_points = partial(n_sections, iterator_type=iforking_point)
n_leaves = partial(n_sections, iterator_type=ileaf)


def map_sections(fun, neurites, neurite_type=NeuriteType.all):
    '''Map fun to all the sections in a collection of neurites'''
    return list(fun(s)
                for s in iter_sections(neurites, neurite_filter=is_type(neurite_type)))


@neurite_fun
def section_branch_orders(neurites, neurite_type=NeuriteType.all):
    '''section branch orders in a collection of neurites'''
    return [branch_order(s)
            for s in iter_sections(neurites, neurite_filter=is_type(neurite_type))]


@neurite_fun
def section_path_lengths(neurites, neurite_type=NeuriteType.all):
    '''Path lengths of a collection of neurites

    Calculates and stores the section lengths in one pass,
    then queries the lengths in the path length iterations.
    This avoids repeatedly calculating the lengths of the
    same sections.
    '''
    dist = {}
    neurite_filter = is_type(neurite_type)

    for s in iter_sections(neurites, neurite_filter=neurite_filter):
        dist[s] = mm.section_length(s.points)

    def pl2(node):
        '''Calculate the path length using cached section lengths'''
        return sum(dist[n] for n in iupstream(node))

    return [pl2(n) for n in iter_sections(neurites, neurite_filter=neurite_filter)]


def segment_lengths(neurites, neurite_type=NeuriteType.all):
    '''Lengths of the segments in a collection of neurites'''
    def _seg_len(sec):
        '''list of segment lengths of a section'''
        vecs = np.diff(sec.points, axis=0)[:, :3]
        return np.sqrt([np.dot(p, p) for p in vecs])

    neurite_filter = is_type(neurite_type)
    return [s for ss in iter_sections(neurites, neurite_filter=neurite_filter)
            for s in _seg_len(ss)]


def segment_midpoints(neurites, neurite_type=NeuriteType.all):
    '''Return a list of segment mid-points in a collection of neurites'''
    def _seg_midpoint(sec):
        '''Return the mid-points of segments in a section'''
        pts = sec.points
        return np.divide(np.add(pts[:-1], pts[1:])[:, :3], 2.0)

    neurite_filter = is_type(neurite_type)
    return [s for ss in iter_sections(neurites, neurite_filter=neurite_filter)
            for s in _seg_midpoint(ss)]


@neurite_fun
def segment_radial_distances(neurites, neurite_type=NeuriteType.all, origin=None):
    '''Lengths of the segments in a collection of neurites'''
    def _seg_rd(sec, pos):
        '''list of radial distances of all segments of a section'''
        mid_pts = np.divide(np.add(sec.points[:-1], sec.points[1:])[:, :3], 2.0)
        return np.sqrt([mm.point_dist2(p, pos) for p in mid_pts])

    neurite_filter = is_type(neurite_type)
    dist = []
    for n in neurites:
        if neurite_filter(n):
            origin = n.root_node.points[0] if origin is None else origin
            dist.extend([s for ss in n.iter_sections() for s in _seg_rd(ss, origin)])

    return dist


@neurite_fun
def local_bifurcation_angles(neurites, neurite_type=NeuriteType.all):
    '''Get a list of local bifurcation angles in a collection of neurites'''
    return [local_bifurcation_angle(b)
            for b in iter_sections(neurites,
                                   iterator_type=ibifurcation_point,
                                   neurite_filter=is_type(neurite_type))]


@neurite_fun
def remote_bifurcation_angles(neurites, neurite_type=NeuriteType.all):
    '''Get a list of remote bifurcation angles in a collection of neurites'''
    return [remote_bifurcation_angle(b)
            for b in iter_sections(neurites,
                                   iterator_type=ibifurcation_point,
                                   neurite_filter=is_type(neurite_type))]


@neurite_fun
def bifurcation_partitions(neurites, neurite_type=NeuriteType.all):
    '''Partition at bifurcation points of a collection of neurites'''
    return [bifurcation_partition(b)
            for b in iter_sections(neurites,
                                   iterator_type=ibifurcation_point,
                                   neurite_filter=is_type(neurite_type))]


@neurite_fun
def section_radial_distances(neurites, neurite_type=NeuriteType.all, origin=None):
    '''Remote bifurcation angles in a collection of neurites'''
    neurite_filter = is_type(neurite_type)
    dist = []
    for n in neurites:
        if neurite_filter(n):
            origin = n.root_node.points[0] if origin is None else origin
            dist.extend([section_radial_distance(s, origin) for s in n.iter_sections()])

    return dist


@neurite_fun
def n_sections_per_neurite(neurites, neurite_type=NeuriteType.all):
    '''Get the number of sections per neurite in a collection of neurites'''
    neurite_filter = is_type(neurite_type)
    return [sum(1 for _ in n.iter_sections()) for n in neurites if neurite_filter(n)]


@neurite_fun
def total_length_per_neurite(neurites, neurite_type=NeuriteType.all):
    '''Get the number of sections per neurite in a collection'''
    neurite_filter = is_type(neurite_type)
    return list(sum(mm.section_length(s.points) for s in n.iter_sections())
                for n in neurites if neurite_filter(n))


@neurite_fun
def principal_direction_extents(neurites, neurite_type=NeuriteType.all, direction=0):
    '''Principal direction extent of neurites in neurons'''
    def _pde(neurite):
        '''Get the PDE of a single neurite'''
        # Get the X, Y,Z coordinates of the points in each section
        points = neurite.points[:, :3]
        return mm.principal_direction_extent(points)[direction]

    neurite_filter = is_type(neurite_type)
    return [_pde(n) for n in neurites if neurite_filter(n)]
