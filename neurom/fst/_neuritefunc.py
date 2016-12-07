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

from itertools import chain

import numpy as np

from neurom import morphmath
from neurom.core import Tree, iter_neurites, iter_sections, NeuriteType
from neurom.core.dataformat import COLS
from neurom.core.types import tree_type_checker as is_type
from neurom.fst import _bifurcationfunc
from neurom.fst import _neuronfunc
from neurom.fst import sectionfunc
from neurom.geom import convex_hull


def total_length(nrn_pop, neurite_type=NeuriteType.all):
    '''Get the total length of all sections in the group of neurons or neurites'''
    nrns = _neuronfunc.neuron_population(nrn_pop)
    return list(sum(section_lengths(n, neurite_type=neurite_type)) for n in nrns)


def n_segments(neurites, neurite_type=NeuriteType.all):
    '''Number of segments in a collection of neurites'''
    return sum(len(s.points) - 1
               for s in iter_sections(neurites, neurite_filter=is_type(neurite_type)))


def n_neurites(neurites, neurite_type=NeuriteType.all):
    '''Number of neurites in a collection of neurites'''
    return sum(1 for _ in iter_neurites(neurites, filt=is_type(neurite_type)))


def n_sections(neurites, neurite_type=NeuriteType.all, iterator_type=Tree.ipreorder):
    '''Number of sections in a collection of neurites'''
    return sum(1 for _ in iter_sections(neurites,
                                        iterator_type=iterator_type,
                                        neurite_filter=is_type(neurite_type)))


def n_bifurcation_points(neurites, neurite_type=NeuriteType.all):
    '''number of bifurcation points in a collection of neurites'''
    return n_sections(neurites, neurite_type=neurite_type, iterator_type=Tree.ibifurcation_point)


def n_forking_points(neurites, neurite_type=NeuriteType.all):
    '''number of forking points in a collection of neurites'''
    return n_sections(neurites, neurite_type=neurite_type, iterator_type=Tree.iforking_point)


def n_leaves(neurites, neurite_type=NeuriteType.all):
    '''number of leaves points in a collection of neurites'''
    return n_sections(neurites, neurite_type=neurite_type, iterator_type=Tree.ileaf)


def map_sections(fun, neurites, neurite_type=NeuriteType.all, iterator_type=Tree.ipreorder):
    '''Map `fun` to all the sections in a collection of neurites'''
    return map(fun, iter_sections(neurites,
                                  iterator_type=iterator_type,
                                  neurite_filter=is_type(neurite_type)))


def section_lengths(neurites, neurite_type=NeuriteType.all):
    '''section lengths in a collection of neurites'''
    def _seclen(sec, **kwargs):
        '''get section length of `sec`'''
        return morphmath.section_length(sec.points, **kwargs)
    return map_sections(_seclen, neurites, neurite_type=neurite_type)


def section_branch_orders(neurites, neurite_type=NeuriteType.all):
    '''section branch orders in a collection of neurites'''
    return map_sections(sectionfunc.branch_order, neurites, neurite_type=neurite_type)


def section_path_lengths(neurites, neurite_type=NeuriteType.all):
    '''Path lengths of a collection of neurites '''
    # Calculates and stores the section lengths in one pass,
    # then queries the lengths in the path length iterations.
    # This avoids repeatedly calculating the lengths of the
    # same sections.
    dist = {}
    neurite_filter = is_type(neurite_type)

    for s in iter_sections(neurites, neurite_filter=neurite_filter):
        dist[s] = s.length

    def pl2(node):
        '''Calculate the path length using cached section lengths'''
        return sum(dist[n] for n in node.iupstream())

    return map_sections(pl2, neurites, neurite_type=neurite_type)


def map_neurons(fun, neurites, neurite_type):
    '''Map `fun` to all the neurites in a single or collection of neurons'''
    nrns = _neuronfunc.neuron_population(neurites)
    return [fun(n, neurite_type=neurite_type) for n in nrns]


def number_of_sections(neurites, neurite_type=NeuriteType.all):
    '''Number of sections in a collection of neurites'''
    return map_neurons(n_sections, neurites, neurite_type)


def number_of_neurites(neurites, neurite_type=NeuriteType.all):
    '''Number of neurites in a collection of neurites'''
    return map_neurons(n_neurites, neurites, neurite_type)


def number_of_bifurcations(neurites, neurite_type=NeuriteType.all):
    '''number of bifurcation points in a collection of neurites'''
    return map_neurons(n_bifurcation_points, neurites, neurite_type)


def number_of_forking_points(neurites, neurite_type=NeuriteType.all):
    '''number of forking points in a collection of neurites'''
    return map_neurons(n_forking_points, neurites, neurite_type)


def number_of_terminations(neurites, neurite_type=NeuriteType.all):
    '''number of leaves points in a collection of neurites'''
    return map_neurons(n_leaves, neurites, neurite_type)


def number_of_segments(neurites, neurite_type=NeuriteType.all):
    '''Number of sections in a collection of neurites'''
    return map_neurons(n_segments, neurites, neurite_type)


def segment_lengths(neurites, neurite_type=NeuriteType.all):
    '''Lengths of the segments in a collection of neurites'''
    def _seg_len(sec):
        '''list of segment lengths of a section'''
        return np.linalg.norm(np.diff(sec.points[:, :COLS.R], axis=0),
                              axis=1)

    neurite_filter = is_type(neurite_type)
    return [s for ss in iter_sections(neurites, neurite_filter=neurite_filter)
            for s in _seg_len(ss)]


def segment_radii(neurites, neurite_type=NeuriteType.all):
    '''arithmetic mean of the radii of the points in segments in a collection of neurites'''
    def _seg_radii(sec):
        '''vectorized mean radii'''
        pts = sec.points[:, COLS.R]
        return np.divide(np.add(pts[:-1], pts[1:]), 2.0)

    neurite_filter = is_type(neurite_type)
    return [s for ss in iter_sections(neurites, neurite_filter=neurite_filter)
            for s in _seg_radii(ss)]


def segment_taper_rates(neurites, neurite_type=NeuriteType.all):
    '''taper rates of the segments in a collection of neurites

    The taper rate is defined as the absolute radii differences divided by length of the section
    '''
    def _seg_taper_rates(sec):
        '''vectorized taper rates'''
        pts = sec.points[:, :COLS.TYPE]
        diff = np.diff(pts, axis=0)
        distance = np.linalg.norm(diff[:, :COLS.R], axis=1)
        return np.divide(2 * np.abs(diff[:, COLS.R]), distance)

    neurite_filter = is_type(neurite_type)
    return [s for ss in iter_sections(neurites, neurite_filter=neurite_filter)
            for s in _seg_taper_rates(ss)]


def segment_meander_angles(neurites, neurite_type=NeuriteType.all):
    '''Inter-segment opening angles in a section'''
    return list(chain.from_iterable(map_sections(
        sectionfunc.section_meander_angles, neurites, neurite_type)))


def segment_midpoints(neurites, neurite_type=NeuriteType.all):
    '''Return a list of segment mid-points in a collection of neurites'''
    def _seg_midpoint(sec):
        '''Return the mid-points of segments in a section'''
        pts = sec.points
        return np.divide(np.add(pts[:-1], pts[1:])[:, :3], 2.0)

    neurite_filter = is_type(neurite_type)
    return [s for ss in iter_sections(neurites, neurite_filter=neurite_filter)
            for s in _seg_midpoint(ss)]


def segment_radial_distances(neurites, neurite_type=NeuriteType.all, origin=None):
    '''Lengths of the segments in a collection of neurites'''
    def _seg_rd(sec, pos):
        '''list of radial distances of all segments of a section'''
        mid_pts = np.divide(np.add(sec.points[:-1], sec.points[1:])[:, :3], 2.0)
        return np.sqrt([morphmath.point_dist2(p, pos) for p in mid_pts])

    dist = []
    for n in iter_neurites(neurites, filt=is_type(neurite_type)):
        pos = n.root_node.points[0] if origin is None else origin
        dist.extend([s for ss in n.iter_sections() for s in _seg_rd(ss, pos)])

    return dist


def local_bifurcation_angles(neurites, neurite_type=NeuriteType.all):
    '''Get a list of local bifurcation angles in a collection of neurites'''
    return map_sections(_bifurcationfunc.local_bifurcation_angle,
                        neurites,
                        neurite_type=neurite_type,
                        iterator_type=Tree.ibifurcation_point)


def remote_bifurcation_angles(neurites, neurite_type=NeuriteType.all):
    '''Get a list of remote bifurcation angles in a collection of neurites'''
    return map_sections(_bifurcationfunc.remote_bifurcation_angle,
                        neurites,
                        neurite_type=neurite_type,
                        iterator_type=Tree.ibifurcation_point)


def bifurcation_partitions(neurites, neurite_type=NeuriteType.all):
    '''Partition at bifurcation points of a collection of neurites'''
    return map(_bifurcationfunc.bifurcation_partition,
               iter_sections(neurites,
                             iterator_type=Tree.ibifurcation_point,
                             neurite_filter=is_type(neurite_type)))


def section_radial_distances(neurites, neurite_type=NeuriteType.all, origin=None):
    '''Remote bifurcation angles in a collection of neurites'''
    dist = []
    for n in iter_neurites(neurites, filt=is_type(neurite_type)):
        pos = n.root_node.points[0] if origin is None else origin
        dist.extend([sectionfunc.section_radial_distance(s, pos)
                     for s in n.iter_sections()])

    return dist


def number_of_sections_per_neurite(neurites, neurite_type=NeuriteType.all):
    '''Get the number of sections per neurite in a collection of neurites'''
    return list(sum(1 for _ in n.iter_sections())
                for n in iter_neurites(neurites, filt=is_type(neurite_type)))


def total_length_per_neurite(neurites, neurite_type=NeuriteType.all):
    '''Get the path length per neurite in a collection'''
    return list(sum(s.length for s in n.iter_sections())
                for n in iter_neurites(neurites, filt=is_type(neurite_type)))


def terminal_path_lengths_per_neurite(neurites, neurite_type=NeuriteType.all):
    '''Get the path lengths to each terminal point per neurite in a collection'''
    return list(sectionfunc.section_path_length(s)
                for n in iter_neurites(neurites, filt=is_type(neurite_type))
                for s in iter_sections(n, iterator_type=Tree.ileaf))


def total_volume_per_neurite(neurites, neurite_type=NeuriteType.all):
    '''Get the volume per neurite in a collection'''
    return list(sum(s.volume for s in n.iter_sections())
                for n in iter_neurites(neurites, filt=is_type(neurite_type)))


def neurite_volume_density(neurites, neurite_type=NeuriteType.all):
    '''Get the volume density per neurite

    The volume density is defined as the ratio of the neurite volume and
    the volume of the neurite's enclosung convex hull
    '''
    def vol_density(neurite):
        '''volume density of a single neurite'''
        return neurite.volume / convex_hull(neurite).volume

    return list(vol_density(n)
                for n in iter_neurites(neurites, filt=is_type(neurite_type)))


def section_volumes(neurites, neurite_type=NeuriteType.all):
    '''section volumes in a collection of neurites'''
    return map_sections(sectionfunc.section_volume, neurites, neurite_type=neurite_type)


def section_areas(neurites, neurite_type=NeuriteType.all):
    '''section areas in a collection of neurites'''
    return map_sections(sectionfunc.section_area, neurites, neurite_type=neurite_type)


def section_tortuosity(neurites, neurite_type=NeuriteType.all):
    '''section tortuosities in a collection of neurites'''
    return map_sections(sectionfunc.section_tortuosity, neurites, neurite_type=neurite_type)


def principal_direction_extents(neurites, neurite_type=NeuriteType.all, direction=0):
    '''Principal direction extent of neurites in neurons'''
    def _pde(neurite):
        '''Get the PDE of a single neurite'''
        # Get the X, Y,Z coordinates of the points in each section
        points = neurite.points[:, :3]
        return morphmath.principal_direction_extent(points)[direction]

    return map(_pde, iter_neurites(neurites, filt=is_type(neurite_type)))
