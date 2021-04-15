# Copyright (c) 2020, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
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

"""Neurite functions."""

import logging
from functools import partial, update_wrapper
from itertools import chain

import numpy as np
import scipy
from neurom import morphmath
from neurom.core import (NeuriteType, Section, iter_neurites, iter_sections,
                         iter_segments)
from neurom.core.dataformat import COLS
from neurom.core.types import tree_type_checker as is_type
from neurom.features import _register_feature, bifurcationfunc, feature, neuronfunc, sectionfunc
from neurom.features.bifurcationfunc import partition_asymmetry
from neurom.features.sectionfunc import downstream_pathlength
from neurom.geom import convex_hull
from neurom.morphmath import interval_lengths

feature = partial(feature, namespace='NEURITEFEATURES')

L = logging.getLogger(__name__)


def _map_sections(fun, neurites, neurite_type=NeuriteType.all, iterator_type=Section.ipreorder):
    """Map `fun` to all the sections in a collection of neurites."""
    return map(fun, iter_sections(neurites,
                                  iterator_type=iterator_type,
                                  neurite_filter=is_type(neurite_type)))


@feature(shape=(...,))
def total_length(nrn_pop, neurite_type=NeuriteType.all):
    """Get the total length of all sections in the group of neurons or neurites."""
    nrns = neuronfunc.neuron_population(nrn_pop)
    return list(sum(section_lengths(n, neurite_type=neurite_type)) for n in nrns)


@feature(shape=())
def max_radial_distance(neurites, neurite_type=NeuriteType.all):
    """Get the maximum radial distances of the termination sections for a collection of neurites."""
    term_radial_distances = section_term_radial_distances(neurites, neurite_type)
    return max(term_radial_distances) if term_radial_distances else 0.


@feature(shape=())
def n_segments(neurites, neurite_type=NeuriteType.all):
    """Number of segments in a collection of neurites."""
    return sum(len(s.points) - 1
               for s in iter_sections(neurites, neurite_filter=is_type(neurite_type)))


@feature(shape=())
def n_neurites(neurites, neurite_type=NeuriteType.all):
    """Number of neurites in a collection of neurites."""
    return sum(1 for _ in iter_neurites(neurites, filt=is_type(neurite_type)))


@feature(shape=())
def n_sections(neurites, neurite_type=NeuriteType.all, iterator_type=Section.ipreorder):
    """Number of sections in a collection of neurites."""
    return sum(1 for _ in iter_sections(neurites,
                                        iterator_type=iterator_type,
                                        neurite_filter=is_type(neurite_type)))


@feature(shape=())
def n_bifurcation_points(neurites, neurite_type=NeuriteType.all):
    """Number of bifurcation points in a collection of neurites."""
    return n_sections(neurites, neurite_type=neurite_type, iterator_type=Section.ibifurcation_point)


@feature(shape=())
def n_forking_points(neurites, neurite_type=NeuriteType.all):
    """Number of forking points in a collection of neurites."""
    return n_sections(neurites, neurite_type=neurite_type, iterator_type=Section.iforking_point)


@feature(shape=())
def n_leaves(neurites, neurite_type=NeuriteType.all):
    """Number of leaves points in a collection of neurites."""
    return n_sections(neurites, neurite_type=neurite_type, iterator_type=Section.ileaf)


@feature(shape=(...,))
def total_area_per_neurite(neurites, neurite_type=NeuriteType.all):
    """Surface area in a collection of neurites.

    The area is defined as the sum of the area of the sections.
    """
    return [neurite.area for neurite in iter_neurites(neurites, filt=is_type(neurite_type))]


def _section_length(section):
    """Get section length of `section`."""
    return morphmath.section_length(section.points)


@feature(shape=(...,))
def section_lengths(neurites, neurite_type=NeuriteType.all):
    """Section lengths in a collection of neurites."""
    return _map_sections(_section_length, neurites, neurite_type=neurite_type)


@feature(shape=(...,))
def section_term_lengths(neurites, neurite_type=NeuriteType.all):
    """Termination section lengths in a collection of neurites."""
    return _map_sections(_section_length, neurites, neurite_type=neurite_type,
                         iterator_type=Section.ileaf)


@feature(shape=(...,))
def section_bif_lengths(neurites, neurite_type=NeuriteType.all):
    """Bifurcation section lengths in a collection of neurites."""
    return _map_sections(_section_length, neurites, neurite_type=neurite_type,
                         iterator_type=Section.ibifurcation_point)


@feature(shape=(...,))
def section_branch_orders(neurites, neurite_type=NeuriteType.all):
    """Section branch orders in a collection of neurites."""
    return _map_sections(sectionfunc.branch_order, neurites, neurite_type=neurite_type)


@feature(shape=(...,))
def section_bif_branch_orders(neurites, neurite_type=NeuriteType.all):
    """Bifurcation section branch orders in a collection of neurites."""
    return _map_sections(sectionfunc.branch_order, neurites, neurite_type=neurite_type,
                         iterator_type=Section.ibifurcation_point)


@feature(shape=(...,))
def section_term_branch_orders(neurites, neurite_type=NeuriteType.all):
    """Termination section branch orders in a collection of neurites."""
    return _map_sections(sectionfunc.branch_order, neurites, neurite_type=neurite_type,
                         iterator_type=Section.ileaf)


@feature(shape=(...,), name='section_path_distances')
def section_path_lengths(neurites, neurite_type=NeuriteType.all):
    """Path lengths of a collection of neurites."""
    # Calculates and stores the section lengths in one pass,
    # then queries the lengths in the path length iterations.
    # This avoids repeatedly calculating the lengths of the
    # same sections.
    dist = {}
    neurite_filter = is_type(neurite_type)

    for s in iter_sections(neurites, neurite_filter=neurite_filter):
        dist[s] = s.length

    def pl2(node):
        """Calculate the path length using cached section lengths."""
        return sum(dist[n] for n in node.iupstream())

    return _map_sections(pl2, neurites, neurite_type=neurite_type)


################################################################################
# Features returning one value per NEURON                                      #
################################################################################

def map_neurons(fun, neurites, neurite_type):
    """Map `fun` to all the neurites in a single or collection of neurons."""
    nrns = neuronfunc.neuron_population(neurites)
    return [fun(n, neurite_type=neurite_type) for n in nrns]


@feature(shape=(...,))
def max_radial_distances(neurites, neurite_type=NeuriteType.all):
    """Get the maximum radial distances of the termination sections for a collection of neurites."""
    return map_neurons(max_radial_distance, neurites, neurite_type)


@feature(shape=(...,))
def number_of_sections(neurites, neurite_type=NeuriteType.all):
    """Number of sections in a collection of neurites."""
    return map_neurons(n_sections, neurites, neurite_type)


@feature(shape=(...,))
def number_of_neurites(neurites, neurite_type=NeuriteType.all):
    """Number of neurites in a collection of neurites."""
    return map_neurons(n_neurites, neurites, neurite_type)


@feature(shape=(...,))
def number_of_bifurcations(neurites, neurite_type=NeuriteType.all):
    """Number of bifurcation points in a collection of neurites."""
    return map_neurons(n_bifurcation_points, neurites, neurite_type)


@feature(shape=(...,))
def number_of_forking_points(neurites, neurite_type=NeuriteType.all):
    """Number of forking points in a collection of neurites."""
    return map_neurons(n_forking_points, neurites, neurite_type)


@feature(shape=(...,))
def number_of_terminations(neurites, neurite_type=NeuriteType.all):
    """Number of leaves points in a collection of neurites."""
    return map_neurons(n_leaves, neurites, neurite_type)


@feature(shape=(...,))
def number_of_segments(neurites, neurite_type=NeuriteType.all):
    """Number of sections in a collection of neurites."""
    return map_neurons(n_segments, neurites, neurite_type)

################################################################################
# Features returning one value per SEGMENT                                     #
################################################################################


def map_segments(func, neurites, neurite_type):
    """Map `func` to all the segments in a collection of neurites.

    `func` accepts a section and returns list of values corresponding to each segment.
    """
    neurite_filter = is_type(neurite_type)
    return [
        s for ss in iter_sections(neurites, neurite_filter=neurite_filter) for s in func(ss)
    ]


@feature(shape=(...,))
def segment_lengths(neurites, neurite_type=NeuriteType.all):
    """Lengths of the segments in a collection of neurites."""
    return map_segments(sectionfunc.segment_lengths, neurites, neurite_type)


@feature(shape=(...,))
def segment_areas(neurites, neurite_type=NeuriteType.all):
    """Areas of the segments in a collection of neurites."""
    return [morphmath.segment_area(seg) for seg
            in iter_segments(neurites, is_type(neurite_type))]


@feature(shape=(...,))
def segment_volumes(neurites, neurite_type=NeuriteType.all):
    """Volumes of the segments in a collection of neurites."""
    def _func(sec):
        """List of segment volumes of a section."""
        return [morphmath.segment_volume(seg) for seg in zip(sec.points[:-1], sec.points[1:])]

    return map_segments(_func, neurites, neurite_type)


@feature(shape=(...,))
def segment_radii(neurites, neurite_type=NeuriteType.all):
    """Arithmetic mean of the radii of the points in segments in a collection of neurites."""
    def _seg_radii(sec):
        """Vectorized mean radii."""
        pts = sec.points[:, COLS.R]
        return np.divide(np.add(pts[:-1], pts[1:]), 2.0)

    return map_segments(_seg_radii, neurites, neurite_type)


@feature(shape=(...,))
def segment_taper_rates(neurites, neurite_type=NeuriteType.all):
    """Diameters taper rates of the segments in a collection of neurites.

    The taper rate is defined as the absolute radii differences divided by length of the section
    """
    def _seg_taper_rates(sec):
        """Vectorized taper rates."""
        pts = sec.points[:, COLS.XYZR]
        diff = np.diff(pts, axis=0)
        distance = np.linalg.norm(diff[:, COLS.XYZ], axis=1)
        return np.divide(2 * np.abs(diff[:, COLS.R]), distance)

    return map_segments(_seg_taper_rates, neurites, neurite_type)


@feature(shape=(...,))
def section_taper_rates(neurites, neurite_type=NeuriteType.all):
    """Diameter taper rates of the sections in a collection of neurites from root to tip.

    Taper rate is defined here as the linear fit along a section.
    It is expected to be negative for neurons.
    """
    def _sec_taper_rate(sec):
        """Taper rate from fit along a section."""
        path_distances = np.cumsum(interval_lengths(sec.points, prepend_zero=True))
        return np.polynomial.polynomial.polyfit(path_distances, 2 * sec.points[:, COLS.R], 1)[1]

    return _map_sections(_sec_taper_rate, neurites, neurite_type=neurite_type)


@feature(shape=(...,))
def segment_meander_angles(neurites, neurite_type=NeuriteType.all):
    """Inter-segment opening angles in a section."""
    return list(chain.from_iterable(_map_sections(
        sectionfunc.section_meander_angles, neurites, neurite_type)))


@feature(shape=(..., 3))
def segment_midpoints(neurites, neurite_type=NeuriteType.all):
    """Return a list of segment mid-points in a collection of neurites."""
    def _seg_midpoint(sec):
        """Return the mid-points of segments in a section."""
        pts = sec.points[:, COLS.XYZ]
        return np.divide(np.add(pts[:-1], pts[1:]), 2.0)

    return map_segments(_seg_midpoint, neurites, neurite_type)


@feature(shape=(...,))
def segment_path_lengths(neurites, neurite_type=NeuriteType.all):
    """Returns pathlengths between all non-root points and their root point."""
    pathlength = {}
    neurite_filter = is_type(neurite_type)

    def _get_pathlength(section):
        if section.id not in pathlength:
            if section.parent:
                pathlength[section.id] = section.parent.length + _get_pathlength(section.parent)
            else:
                pathlength[section.id] = 0
        return pathlength[section.id]

    result = [_get_pathlength(section) + np.cumsum(sectionfunc.segment_lengths(section))
              for section in iter_sections(neurites, neurite_filter=neurite_filter)]
    return np.hstack(result) if result else np.array([])


@feature(shape=(...,))
def segment_radial_distances(neurites, neurite_type=NeuriteType.all, origin=None):
    """Returns the list of distances between all segment mid points and origin."""
    def _radial_distances(sec, pos):
        """List of distances between the mid point of each segment and pos."""
        mid_pts = 0.5 * (sec.points[:-1, COLS.XYZ] + sec.points[1:, COLS.XYZ])
        return np.linalg.norm(mid_pts - pos[COLS.XYZ], axis=1)

    dist = []
    for n in iter_neurites(neurites, filt=is_type(neurite_type)):
        pos = n.root_node.points[0] if origin is None else origin
        dist.extend([s for ss in n.iter_sections() for s in _radial_distances(ss, pos)])

    return dist


@feature(shape=(...,))
def local_bifurcation_angles(neurites, neurite_type=NeuriteType.all):
    """Get a list of local bifurcation angles in a collection of neurites."""
    return _map_sections(bifurcationfunc.local_bifurcation_angle,
                         neurites,
                         neurite_type=neurite_type,
                         iterator_type=Section.ibifurcation_point)


@feature(shape=(...,))
def remote_bifurcation_angles(neurites, neurite_type=NeuriteType.all):
    """Get a list of remote bifurcation angles in a collection of neurites."""
    return _map_sections(bifurcationfunc.remote_bifurcation_angle,
                         neurites,
                         neurite_type=neurite_type,
                         iterator_type=Section.ibifurcation_point)


@feature(shape=(...,), name='partition')
def bifurcation_partitions(neurites, neurite_type=NeuriteType.all):
    """Partition at bifurcation points of a collection of neurites."""
    return map(bifurcationfunc.bifurcation_partition,
               iter_sections(neurites,
                             iterator_type=Section.ibifurcation_point,
                             neurite_filter=is_type(neurite_type)))


@feature(shape=(...,), name='partition_asymmetry')
def partition_asymmetries(neurites, neurite_type=NeuriteType.all, variant='branch-order'):
    """Partition asymmetry at bifurcation points of a collection of neurites.

    Variant: length is a different definition, as the absolute difference in
    downstream path lenghts, relative to the total neurite path length
    """
    if variant not in {'branch-order', 'length'}:
        raise ValueError('Please provide a valid variant for partition asymmetry,\
                         found %s' % variant)

    if variant == 'branch-order':
        return map(partition_asymmetry,
                   iter_sections(neurites,
                                 iterator_type=Section.ibifurcation_point,
                                 neurite_filter=is_type(neurite_type)))

    asymmetries = list()
    for neurite in iter_neurites(neurites, filt=is_type(neurite_type)):
        neurite_length = total_length_per_neurite(neurite)[0]
        for section in iter_sections(neurite,
                                     iterator_type=Section.ibifurcation_point,
                                     neurite_filter=is_type(neurite_type)):
            pathlength_diff = abs(downstream_pathlength(section.children[0]) -
                                  downstream_pathlength(section.children[1]))
            asymmetries.append(pathlength_diff / neurite_length)
    return asymmetries


# Register `partition_asymmetries` variant
_partition_asymmetry_length = partial(partition_asymmetries, variant='length')
update_wrapper(_partition_asymmetry_length, partition_asymmetries)  # this fixes the docstring
_register_feature('NEURITEFEATURES', 'partition_asymmetry_length',
                  _partition_asymmetry_length, shape=(...,))


@feature(shape=(...,))
def sibling_ratios(neurites, neurite_type=NeuriteType.all, method='first'):
    """Sibling ratios at bifurcation points of a collection of neurites.

    The sibling ratio is the ratio between the diameters of the
    smallest and the largest child. It is a real number between
    0 and 1. Method argument allows one to consider mean diameters
    along the child section instead of diameter of the first point.
    """
    return map(lambda bif_point: bifurcationfunc.sibling_ratio(bif_point, method),
               iter_sections(neurites,
                             iterator_type=Section.ibifurcation_point,
                             neurite_filter=is_type(neurite_type)))


@feature(shape=(..., 2))
def partition_pairs(neurites, neurite_type=NeuriteType.all):
    """Partition pairs at bifurcation points of a collection of neurites.

    Partition pair is defined as the number of bifurcations at the two
    daughters of the bifurcating section
    """
    return map(bifurcationfunc.partition_pair,
               iter_sections(neurites,
                             iterator_type=Section.ibifurcation_point,
                             neurite_filter=is_type(neurite_type)))


@feature(shape=(...,))
def diameter_power_relations(neurites, neurite_type=NeuriteType.all, method='first'):
    """Calculate the diameter power relation at a bifurcation point.

    Diameter power relation is defined in https://www.ncbi.nlm.nih.gov/pubmed/18568015

    This quantity gives an indication of how far the branching is from
    the Rall ratio (when =1).
    """
    return (bifurcationfunc.diameter_power_relation(bif_point, method)
            for bif_point in iter_sections(neurites,
                                           iterator_type=Section.ibifurcation_point,
                                           neurite_filter=is_type(neurite_type)))


@feature(shape=(...,))
def section_radial_distances(neurites, neurite_type=NeuriteType.all, origin=None,
                             iterator_type=Section.ipreorder):
    """Section radial distances in a collection of neurites.

    The iterator_type can be used to select only terminal sections (ileaf)
    or only bifurcations (ibifurcation_point).
    """
    dist = []
    for n in iter_neurites(neurites, filt=is_type(neurite_type)):
        pos = n.root_node.points[0] if origin is None else origin
        dist.extend(sectionfunc.section_radial_distance(s, pos)
                    for s in iter_sections(n,
                                           iterator_type=iterator_type))
    return dist


@feature(shape=(...,))
def section_term_radial_distances(neurites, neurite_type=NeuriteType.all, origin=None):
    """Get the radial distances of the termination sections for a collection of neurites."""
    return section_radial_distances(neurites, neurite_type=neurite_type, origin=origin,
                                    iterator_type=Section.ileaf)


@feature(shape=(...,))
def section_bif_radial_distances(neurites, neurite_type=NeuriteType.all, origin=None):
    """Get the radial distances of the bifurcation sections for a collection of neurites."""
    return section_radial_distances(neurites, neurite_type=neurite_type, origin=origin,
                                    iterator_type=Section.ibifurcation_point)


@feature(shape=(...,))
def number_of_sections_per_neurite(neurites, neurite_type=NeuriteType.all):
    """Get the number of sections per neurite in a collection of neurites."""
    return list(sum(1 for _ in n.iter_sections())
                for n in iter_neurites(neurites, filt=is_type(neurite_type)))


@feature(shape=(...,))
def total_length_per_neurite(neurites, neurite_type=NeuriteType.all):
    """Get the path length per neurite in a collection."""
    return list(sum(s.length for s in n.iter_sections())
                for n in iter_neurites(neurites, filt=is_type(neurite_type)))


@feature(shape=(...,))
def neurite_lengths(neurites, neurite_type=NeuriteType.all):
    """Get the path length per neurite in a collection."""
    return total_length_per_neurite(neurites, neurite_type)


@feature(shape=(...,))
def terminal_path_lengths_per_neurite(neurites, neurite_type=NeuriteType.all):
    """Get the path lengths to each terminal point per neurite in a collection."""
    return list(sectionfunc.section_path_length(s)
                for n in iter_neurites(neurites, filt=is_type(neurite_type))
                for s in iter_sections(n, iterator_type=Section.ileaf))


@feature(shape=(...,), name='neurite_volumes')
def total_volume_per_neurite(neurites, neurite_type=NeuriteType.all):
    """Get the volume per neurite in a collection."""
    return list(sum(s.volume for s in n.iter_sections())
                for n in iter_neurites(neurites, filt=is_type(neurite_type)))


@feature(shape=(...,))
def neurite_volume_density(neurites, neurite_type=NeuriteType.all):
    """Get the volume density per neurite.

    The volume density is defined as the ratio of the neurite volume and
    the volume of the neurite's enclosing convex hull

    TODO: the convex hull fails on some morphologies, it may be good to instead use
        bounding_box to compute the neurite enclosing volume

    .. note:: Returns `np.nan` if the convex hull computation fails.
    """
    def vol_density(neurite):
        """Volume density of a single neurite."""
        try:
            volume = convex_hull(neurite).volume
        except scipy.spatial.qhull.QhullError:
            L.exception('Failure to compute neurite volume using the convex hull. '
                        'Feature `neurite_volume_density` will return `np.nan`.\n')
            return np.nan

        return neurite.volume / volume

    return list(vol_density(n)
                for n in iter_neurites(neurites, filt=is_type(neurite_type)))


@feature(shape=(...,))
def section_volumes(neurites, neurite_type=NeuriteType.all):
    """Section volumes in a collection of neurites."""
    return _map_sections(sectionfunc.section_volume, neurites, neurite_type=neurite_type)


@feature(shape=(...,))
def section_areas(neurites, neurite_type=NeuriteType.all):
    """Section areas in a collection of neurites."""
    return _map_sections(sectionfunc.section_area, neurites, neurite_type=neurite_type)


@feature(shape=(...,))
def section_tortuosity(neurites, neurite_type=NeuriteType.all):
    """Section tortuosities in a collection of neurites."""
    return _map_sections(sectionfunc.section_tortuosity, neurites, neurite_type=neurite_type)


@feature(shape=(...,))
def section_end_distances(neurites, neurite_type=NeuriteType.all):
    """Section end to end distances in a collection of neurites."""
    return _map_sections(sectionfunc.section_end_distance, neurites, neurite_type=neurite_type)


@feature(shape=(...,))
def principal_direction_extents(neurites, neurite_type=NeuriteType.all, direction=0):
    """Principal direction extent of neurites in neurons."""
    def _pde(neurite):
        """Get the PDE of a single neurite."""
        # Get the X, Y,Z coordinates of the points in each section
        points = neurite.points[:, :3]
        return morphmath.principal_direction_extent(points)[direction]

    return [_pde(neurite) for neurite in iter_neurites(neurites, filt=is_type(neurite_type))]


@feature(shape=(...,))
def section_strahler_orders(neurites, neurite_type=NeuriteType.all):
    """Inter-segment opening angles in a section."""
    return _map_sections(sectionfunc.strahler_order, neurites, neurite_type)
