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

"""Neurite features.

Any public function from this namespace can be called via the features mechanism. If calling
directly the function in this namespace can only accept a neurite as its input. If you want to
apply it to anything other than neurite then you must use the features mechanism e.g.
``features.get``.

>>> import neurom
>>> from neurom import features
>>> m = neurom.load_morphology('path/to/morphology')
>>> features.get('max_radial_distance', m.neurites[0])
>>> features.get('max_radial_distance', m)
>>> features.get('number_of_segments', m.neurites, neurite_type=neurom.AXON)

For more details see :ref:`features`.
"""

import logging
from functools import partial
from itertools import chain

import numpy as np
import scipy
from neurom import morphmath
from neurom.core.morphology import Section
from neurom.core.dataformat import COLS
from neurom.features import NameSpace, feature, bifurcation as bf, section as sf
from neurom.geom import convex_hull
from neurom.morphmath import interval_lengths

feature = partial(feature, namespace=NameSpace.NEURITE)

L = logging.getLogger(__name__)


def _map_sections(fun, neurite, iterator_type=Section.ipreorder):
    """Map `fun` to all the sections."""
    return list(map(fun, (s for s in iterator_type(neurite.root_node))))


@feature(shape=())
def max_radial_distance(neurite):
    """Get the maximum radial distances of the termination sections."""
    term_radial_distances = section_term_radial_distances(neurite)
    return max(term_radial_distances) if term_radial_distances else 0.


@feature(shape=())
def number_of_segments(neurite):
    """Number of segments."""
    return sum(len(s.points) - 1 for s in Section.ipreorder(neurite.root_node))


@feature(shape=())
def number_of_sections(neurite, iterator_type=Section.ipreorder):
    """Number of sections. For a morphology it will be a sum of all neurites sections numbers."""
    return sum(1 for _ in iterator_type(neurite.root_node))


@feature(shape=())
def number_of_bifurcations(neurite):
    """Number of bf points."""
    return number_of_sections(neurite, iterator_type=Section.ibifurcation_point)


@feature(shape=())
def number_of_forking_points(neurite):
    """Number of forking points."""
    return number_of_sections(neurite, iterator_type=Section.iforking_point)


@feature(shape=())
def number_of_leaves(neurite):
    """Number of leaves points."""
    return number_of_sections(neurite, iterator_type=Section.ileaf)


@feature(shape=())
def total_length(neurite):
    """Neurite length. For a morphology it will be a sum of all neurite lengths."""
    return sum(s.length for s in neurite.iter_sections())


@feature(shape=())
def total_area(neurite):
    """Neurite surface area. For a morphology it will be a sum of all neurite areas.

    The area is defined as the sum of the area of the sections.
    """
    return neurite.area


@feature(shape=())
def total_volume(neurite):
    """Neurite volume. For a morphology it will be a sum of neurites volumes."""
    return sum(s.volume for s in Section.ipreorder(neurite.root_node))


def _section_length(section):
    """Get section length of `section`."""
    return morphmath.section_length(section.points)


@feature(shape=(...,))
def section_lengths(neurite):
    """Section lengths."""
    return _map_sections(_section_length, neurite)


@feature(shape=(...,))
def section_term_lengths(neurite):
    """Termination section lengths."""
    return _map_sections(_section_length, neurite, Section.ileaf)


@feature(shape=(...,))
def section_bif_lengths(neurite):
    """Bifurcation section lengths."""
    return _map_sections(_section_length, neurite, Section.ibifurcation_point)


@feature(shape=(...,))
def section_branch_orders(neurite):
    """Section branch orders."""
    return _map_sections(sf.branch_order, neurite)


@feature(shape=(...,))
def section_bif_branch_orders(neurite):
    """Bifurcation section branch orders."""
    return _map_sections(sf.branch_order, neurite, Section.ibifurcation_point)


@feature(shape=(...,))
def section_term_branch_orders(neurite):
    """Termination section branch orders."""
    return _map_sections(sf.branch_order, neurite, Section.ileaf)


@feature(shape=(...,))
def section_path_distances(neurite):
    """Path lengths."""

    def pl2(node):
        """Calculate the path length using cached section lengths."""
        return sum(n.length for n in node.iupstream())

    return _map_sections(pl2, neurite)


################################################################################
# Features returning one value per SEGMENT                                     #
################################################################################


def _map_segments(func, neurite):
    """Map `func` to all the segments.

    `func` accepts a section and returns list of values corresponding to each segment.
    """
    tmp = [mapped_seg for s in Section.ipreorder(neurite.root_node) for mapped_seg in func(s)]
    return tmp


@feature(shape=(...,))
def segment_lengths(neurite):
    """Lengths of the segments."""
    return _map_segments(sf.segment_lengths, neurite)


@feature(shape=(...,))
def segment_areas(neurite):
    """Areas of the segments."""
    return [morphmath.segment_area(seg)
            for s in Section.ipreorder(neurite.root_node)
            for seg in zip(s.points[:-1], s.points[1:])]


@feature(shape=(...,))
def segment_volumes(neurite):
    """Volumes of the segments."""

    def _func(sec):
        """List of segment volumes of a section."""
        return [morphmath.segment_volume(seg) for seg in zip(sec.points[:-1], sec.points[1:])]

    return _map_segments(_func, neurite)


@feature(shape=(...,))
def segment_radii(neurite):
    """Arithmetic mean of the radii of the points in segments."""

    def _seg_radii(sec):
        """Vectorized mean radii."""
        pts = sec.points[:, COLS.R]
        return np.divide(np.add(pts[:-1], pts[1:]), 2.0)

    return _map_segments(_seg_radii, neurite)


@feature(shape=(...,))
def segment_taper_rates(neurite):
    """Diameters taper rates of the segments.

    The taper rate is defined as the absolute radii differences divided by length of the section
    """

    def _seg_taper_rates(sec):
        """Vectorized taper rates."""
        pts = sec.points[:, COLS.XYZR]
        diff = np.diff(pts, axis=0)
        distance = np.linalg.norm(diff[:, COLS.XYZ], axis=1)
        return np.divide(2 * np.abs(diff[:, COLS.R]), distance)

    return _map_segments(_seg_taper_rates, neurite)


@feature(shape=(...,))
def section_taper_rates(neurite):
    """Diameter taper rates of the sections from root to tip.

    Taper rate is defined here as the linear fit along a section.
    It is expected to be negative for morphologies.
    """

    def _sec_taper_rate(sec):
        """Taper rate from fit along a section."""
        path_distances = np.cumsum(interval_lengths(sec.points, prepend_zero=True))
        return np.polynomial.polynomial.polyfit(path_distances, 2 * sec.points[:, COLS.R], 1)[1]

    return _map_sections(_sec_taper_rate, neurite)


@feature(shape=(...,))
def segment_meander_angles(neurite):
    """Inter-segment opening angles in a section."""
    return list(chain.from_iterable(_map_sections(sf.section_meander_angles, neurite)))


@feature(shape=(..., 3))
def segment_midpoints(neurite):
    """Return a list of segment mid-points."""

    def _seg_midpoint(sec):
        """Return the mid-points of segments in a section."""
        pts = sec.points[:, COLS.XYZ]
        return np.divide(np.add(pts[:-1], pts[1:]), 2.0)

    return _map_segments(_seg_midpoint, neurite)


@feature(shape=(...,))
def segment_path_lengths(neurite):
    """Returns pathlengths between all non-root points and their root point."""
    pathlength = {}

    def _get_pathlength(section):
        if section.id not in pathlength:
            if section.parent:
                pathlength[section.id] = section.parent.length + _get_pathlength(section.parent)
            else:
                pathlength[section.id] = 0
        return pathlength[section.id]

    result = [_get_pathlength(section) + np.cumsum(sf.segment_lengths(section))
              for section in Section.ipreorder(neurite.root_node)]
    return np.hstack(result).tolist() if result else []


@feature(shape=(...,))
def segment_radial_distances(neurite, origin=None):
    """Returns the list of distances between all segment mid points and origin."""

    def _radial_distances(sec, pos):
        """List of distances between the mid point of each segment and pos."""
        mid_pts = 0.5 * (sec.points[:-1, COLS.XYZ] + sec.points[1:, COLS.XYZ])
        return np.linalg.norm(mid_pts - pos[COLS.XYZ], axis=1)

    pos = neurite.root_node.points[0] if origin is None else origin
    # return [s for ss in n.iter_sections() for s in _radial_distances(ss, pos)]
    return [d for s in Section.ipreorder(neurite.root_node) for d in _radial_distances(s, pos)]


@feature(shape=(...,))
def local_bifurcation_angles(neurite):
    """Get a list of local bf angles."""
    return _map_sections(bf.local_bifurcation_angle,
                         neurite,
                         iterator_type=Section.ibifurcation_point)


@feature(shape=(...,))
def remote_bifurcation_angles(neurite):
    """Get a list of remote bf angles."""
    return _map_sections(bf.remote_bifurcation_angle,
                         neurite,
                         iterator_type=Section.ibifurcation_point)


@feature(shape=(...,))
def partition_asymmetry(neurite, variant='branch-order', method='petilla'):
    """Partition asymmetry at bf points.

    Variant: length is a different definition, as the absolute difference in
    downstream path lenghts, relative to the total neurite path length
    Method: 'petilla' or 'uylings'. The former is default. The latter uses ``-2`` shift. See
    :func:`neurom.features.bifurcationfunc.partition_asymmetry`
    """
    if variant not in {'branch-order', 'length'}:
        raise ValueError('Please provide a valid variant for partition asymmetry,'
                         f'found {variant}')
    if method not in {'petilla', 'uylings'}:
        raise ValueError('Please provide a valid method for partition asymmetry,'
                         'either "petilla" or "uylings"')

    if variant == 'branch-order':
        return _map_sections(
            partial(bf.partition_asymmetry, uylings=method == 'uylings'),
            neurite,
            Section.ibifurcation_point)

    asymmetries = []
    neurite_length = total_length(neurite)
    for section in Section.ibifurcation_point(neurite.root_node):
        pathlength_diff = abs(sf.downstream_pathlength(section.children[0]) -
                              sf.downstream_pathlength(section.children[1]))
        asymmetries.append(pathlength_diff / neurite_length)
    return asymmetries


@feature(shape=(...,))
def partition_asymmetry_length(neurite, method='petilla'):
    """'partition_asymmetry' feature with `variant='length'`.

    Because it is often used, it has a dedicated feature.
    """
    return partition_asymmetry(neurite, 'length', method)


@feature(shape=(...,))
def bifurcation_partitions(neurite):
    """Partition at bf points."""
    return _map_sections(bf.bifurcation_partition,
                         neurite,
                         Section.ibifurcation_point)


@feature(shape=(...,))
def sibling_ratios(neurite, method='first'):
    """Sibling ratios at bf points.

    The sibling ratio is the ratio between the diameters of the
    smallest and the largest child. It is a real number between
    0 and 1. Method argument allows one to consider mean diameters
    along the child section instead of diameter of the first point.
    """
    return _map_sections(partial(bf.sibling_ratio, method=method),
                         neurite,
                         Section.ibifurcation_point)


@feature(shape=(..., 2))
def partition_pairs(neurite):
    """Partition pairs at bf points.

    Partition pair is defined as the number of bifurcations at the two
    daughters of the bifurcating section
    """
    return _map_sections(bf.partition_pair,
                         neurite,
                         Section.ibifurcation_point)


@feature(shape=(...,))
def diameter_power_relations(neurite, method='first'):
    """Calculate the diameter power relation at a bf point.

    Diameter power relation is defined in https://www.ncbi.nlm.nih.gov/pubmed/18568015

    This quantity gives an indication of how far the branching is from
    the Rall ratio (when =1).
    """
    return _map_sections(partial(bf.diameter_power_relation, method=method),
                         neurite,
                         Section.ibifurcation_point)


@feature(shape=(...,))
def section_radial_distances(neurite, origin=None, iterator_type=Section.ipreorder):
    """Section radial distances.

    The iterator_type can be used to select only terminal sections (ileaf)
    or only bifurcations (ibifurcation_point).
    """
    pos = neurite.root_node.points[0] if origin is None else origin
    return _map_sections(partial(sf.section_radial_distance, origin=pos),
                         neurite,
                         iterator_type)


@feature(shape=(...,))
def section_term_radial_distances(neurite, origin=None):
    """Get the radial distances of the termination sections."""
    return section_radial_distances(neurite, origin, Section.ileaf)


@feature(shape=(...,))
def section_bif_radial_distances(neurite, origin=None):
    """Get the radial distances of the bf sections."""
    return section_radial_distances(neurite, origin, Section.ibifurcation_point)


@feature(shape=(...,))
def terminal_path_lengths(neurite):
    """Get the path lengths to each terminal point."""
    return _map_sections(sf.section_path_length, neurite, Section.ileaf)


@feature(shape=())
def volume_density(neurite):
    """Get the volume density.

    The volume density is defined as the ratio of the neurite volume and
    the volume of the neurite's enclosing convex hull

    TODO: convex hull fails on some morphologies, it may be good to instead use
        bounding_box to compute the neurite enclosing volume

    .. note:: Returns `np.nan` if the convex hull computation fails.
    """
    try:
        volume = convex_hull(neurite).volume
    except scipy.spatial.qhull.QhullError:
        L.exception('Failure to compute neurite volume using the convex hull. '
                    'Feature `volume_density` will return `np.nan`.\n')
        return np.nan

    return neurite.volume / volume


@feature(shape=(...,))
def section_volumes(neurite):
    """Section volumes."""
    return _map_sections(sf.section_volume, neurite)


@feature(shape=(...,))
def section_areas(neurite):
    """Section areas."""
    return _map_sections(sf.section_area, neurite)


@feature(shape=(...,))
def section_tortuosity(neurite):
    """Section tortuosities."""
    return _map_sections(sf.section_tortuosity, neurite)


@feature(shape=(...,))
def section_end_distances(neurite):
    """Section end to end distances."""
    return _map_sections(sf.section_end_distance, neurite)


@feature(shape=(...,))
def principal_direction_extents(neurite, direction=0):
    """Principal direction extent of neurites in morphologies."""
    points = neurite.points[:, :3]
    return [morphmath.principal_direction_extent(points)[direction]]


@feature(shape=(...,))
def section_strahler_orders(neurite):
    """Inter-segment opening angles in a section."""
    return _map_sections(sf.strahler_order, neurite)
