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
>>> m = neurom.load_morphology("tests/data/swc/Neuron.swc")
>>> max_radial_distances1 = features.get('max_radial_distance', m.neurites)
>>> max_radial_distances2 = features.get('max_radial_distance', m.neurites[0])
>>> max_radial_distances3 = features.get('max_radial_distance', m)
>>> n_segments = features.get('number_of_segments', m, neurite_type=neurom.AXON)

For more details see :ref:`features`.
"""

import logging
from functools import partial

import numpy as np

from neurom import morphmath, utils
from neurom.core.dataformat import COLS
from neurom.core.morphology import Section, iter_points
from neurom.core.types import NeuriteType, is_composite_type
from neurom.core.types import tree_type_checker as is_type
from neurom.features import NameSpace
from neurom.features import bifurcation as bf
from neurom.features import feature
from neurom.features import section as sf
from neurom.morphmath import convex_hull

feature = partial(feature, namespace=NameSpace.NEURITE)

L = logging.getLogger(__name__)


def _map_sections(fun, neurite, iterator_type=Section.ipreorder, section_type=NeuriteType.all):
    """Map `fun` to all the sections."""
    check_type = is_type(section_type)

    if (
        section_type != NeuriteType.all
        and not any(is_composite_type(i) for i in check_type.type)
        and iterator_type in {Section.ibifurcation_point, Section.iforking_point}
    ):

        def filt(section):
            return check_type(section) and Section.is_homogeneous_point(section)

    else:
        filt = check_type

    return list(map(fun, filter(filt, iterator_type(neurite.root_node))))


@feature(shape=())
def number_of_segments(neurite, section_type=NeuriteType.all):
    """Number of segments."""
    return sum(_map_sections(sf.number_of_segments, neurite, section_type=section_type))


@feature(shape=())
def number_of_sections(neurite, iterator_type=Section.ipreorder, section_type=NeuriteType.all):
    """Number of sections. For a morphology it will be a sum of all neurites sections numbers."""
    return len(
        _map_sections(lambda x: 1, neurite, iterator_type=iterator_type, section_type=section_type)
    )


@feature(shape=())
def number_of_bifurcations(neurite, section_type=NeuriteType.all):
    """Number of bf points."""
    return number_of_sections(
        neurite, iterator_type=Section.ibifurcation_point, section_type=section_type
    )


@feature(shape=())
def number_of_forking_points(neurite, section_type=NeuriteType.all):
    """Number of forking points."""
    return number_of_sections(
        neurite, iterator_type=Section.iforking_point, section_type=section_type
    )


@feature(shape=())
def number_of_leaves(neurite, section_type=NeuriteType.all):
    """Number of leaves points."""
    return number_of_sections(neurite, iterator_type=Section.ileaf, section_type=section_type)


@feature(shape=())
def total_length(neurite, section_type=NeuriteType.all):
    """Neurite length. For a morphology it will be a sum of all neurite lengths."""
    return sum(_map_sections(sf.section_length, neurite, section_type=section_type))


@feature(shape=())
def total_area(neurite, section_type=NeuriteType.all):
    """Neurite surface area. For a morphology it will be a sum of all neurite areas.

    The area is defined as the sum of the area of the sections.
    """
    return sum(_map_sections(sf.section_area, neurite, section_type=section_type))


@feature(shape=())
def total_volume(neurite, section_type=NeuriteType.all):
    """Neurite volume. For a morphology it will be a sum of neurites volumes."""
    return sum(_map_sections(sf.section_volume, neurite, section_type=section_type))


@feature(shape=(...,))
def section_lengths(neurite, section_type=NeuriteType.all):
    """Section lengths."""
    return _map_sections(sf.section_length, neurite, section_type=section_type)


@feature(shape=(...,))
def section_term_lengths(neurite, section_type=NeuriteType.all):
    """Termination section lengths."""
    return _map_sections(sf.section_length, neurite, Section.ileaf, section_type)


@feature(shape=(...,))
def section_bif_lengths(neurite, section_type=NeuriteType.all):
    """Bifurcation section lengths."""
    return _map_sections(sf.section_length, neurite, Section.ibifurcation_point, section_type)


@feature(shape=(...,))
def section_branch_orders(neurite, section_type=NeuriteType.all):
    """Section branch orders."""
    return _map_sections(sf.branch_order, neurite, section_type=section_type)


@feature(shape=(...,))
def section_bif_branch_orders(neurite, section_type=NeuriteType.all):
    """Bifurcation section branch orders."""
    return _map_sections(
        sf.branch_order, neurite, Section.ibifurcation_point, section_type=section_type
    )


@feature(shape=(...,))
def section_term_branch_orders(neurite, section_type=NeuriteType.all):
    """Termination section branch orders."""
    return _map_sections(sf.branch_order, neurite, Section.ileaf, section_type=section_type)


@feature(shape=(...,))
def section_path_distances(neurite, iterator_type=Section.ipreorder, section_type=NeuriteType.all):
    """Path lengths."""
    return _map_sections(
        partial(sf.section_path_length, stop_node=neurite.root_node),
        neurite,
        iterator_type=iterator_type,
        section_type=section_type,
    )


################################################################################
# Features returning one value per SEGMENT                                     #
################################################################################


def _map_segments(func, neurite, section_type=NeuriteType.all):
    """Map `func` to all the segments.

    `func` accepts a section and returns list of values corresponding to each segment.
    """
    return list(utils.flatten(_map_sections(func, neurite, section_type=section_type)))


@feature(shape=(...,))
def segment_lengths(neurite, section_type=NeuriteType.all):
    """Lengths of the segments."""
    return _map_segments(sf.segment_lengths, neurite, section_type=section_type)


@feature(shape=(...,))
def segment_areas(neurite, section_type=NeuriteType.all):
    """Areas of the segments."""
    return _map_segments(sf.segment_areas, neurite, section_type=section_type)


@feature(shape=(...,))
def segment_volumes(neurite, section_type=NeuriteType.all):
    """Volumes of the segments."""
    return _map_segments(sf.segment_volumes, neurite, section_type=section_type)


@feature(shape=(...,))
def segment_radii(neurite, section_type=NeuriteType.all):
    """Arithmetic mean of the radii of the points in segments."""
    return _map_segments(sf.segment_mean_radii, neurite, section_type=section_type)


@feature(shape=(...,))
def segment_taper_rates(neurite, section_type=NeuriteType.all):
    """Diameters taper rates of the segments.

    The taper rate is defined as the absolute radii differences divided by length of the section
    """
    return _map_segments(sf.segment_taper_rates, neurite, section_type=section_type)


@feature(shape=(...,))
def section_taper_rates(neurite, section_type=NeuriteType.all):
    """Diameter taper rates of the sections from root to tip.

    Taper rate is defined here as the linear fit along a section.
    It is expected to be negative for morphologies.
    """
    return _map_sections(sf.taper_rate, neurite, section_type=section_type)


@feature(shape=(...,))
def segment_meander_angles(neurite, section_type=NeuriteType.all):
    """Inter-segment opening angles in a section."""
    return _map_segments(sf.section_meander_angles, neurite, section_type=section_type)


@feature(shape=(..., 3))
def segment_midpoints(neurite, section_type=NeuriteType.all):
    """Return a list of segment mid-points."""
    return _map_segments(sf.segment_midpoints, neurite, section_type=section_type)


@feature(shape=(...,))
def segment_path_lengths(neurite, section_type=NeuriteType.all):
    """Returns pathlengths between all non-root points and their root point."""
    pathlength = {}

    def segments_path_length(section):
        if section.id not in pathlength:
            pathlength[section.id] = (
                0.0
                if section.id == neurite.root_node.id
                else section.parent.length + pathlength[section.parent.id]
            )

        return pathlength[section.id] + np.cumsum(sf.segment_lengths(section))

    return _map_segments(segments_path_length, neurite, section_type=section_type)


@feature(shape=(...,))
def segment_radial_distances(neurite, origin=None, section_type=NeuriteType.all):
    """Returns the list of distances between all segment mid points and origin."""
    origin = neurite.root_node.points[0, COLS.XYZ] if origin is None else origin
    return _map_segments(
        func=partial(sf.segment_midpoint_radial_distances, origin=origin),
        neurite=neurite,
        section_type=section_type,
    )


@feature(shape=(...,))
def local_bifurcation_angles(neurite, section_type=NeuriteType.all):
    """Get a list of local bf angles."""
    return _map_sections(
        bf.local_bifurcation_angle,
        neurite,
        iterator_type=Section.ibifurcation_point,
        section_type=section_type,
    )


@feature(shape=(...,))
def remote_bifurcation_angles(neurite, section_type=NeuriteType.all):
    """Get a list of remote bf angles."""
    return _map_sections(
        bf.remote_bifurcation_angle,
        neurite,
        iterator_type=Section.ibifurcation_point,
        section_type=section_type,
    )


@feature(shape=(...,))
def partition_asymmetry(
    neurite, variant='branch-order', method='petilla', section_type=NeuriteType.all
):
    """Partition asymmetry at bf points.

    Variant: length is a different definition, as the absolute difference in
    downstream path lenghts, relative to the total neurite path length
    Method: 'petilla' or 'uylings'. The former is default. The latter uses ``-2`` shift.
    """
    if variant not in {'branch-order', 'length'}:
        raise ValueError(
            "Please provide a valid variant for partition asymmetry. "
            f"Expected 'branch-order' or 'length', got {variant}."
        )
    if method not in {'petilla', 'uylings'}:
        raise ValueError(
            "Please provide a valid method for partition asymmetry. "
            f"Expected 'petilla' or 'uylings', got {method}."
        )

    # create a downstream iterator that is filtered by the section type
    it_type = utils.filtered_iterator(is_type(section_type), Section.ipreorder)

    if variant == 'branch-order':
        return _map_sections(
            partial(bf.partition_asymmetry, uylings=method == 'uylings', iterator_type=it_type),
            neurite,
            iterator_type=Section.ibifurcation_point,
            section_type=section_type,
        )

    return _map_sections(
        partial(
            bf.downstream_pathlength_asymmetry,
            normalization_length=total_length(neurite, section_type=section_type),
            iterator_type=it_type,
        ),
        neurite,
        iterator_type=Section.ibifurcation_point,
        section_type=section_type,
    )


@feature(shape=(...,))
def partition_asymmetry_length(neurite, method='petilla', section_type=NeuriteType.all):
    """'partition_asymmetry' feature with `variant='length'`.

    Because it is often used, it has a dedicated feature.
    """
    return partition_asymmetry(neurite, 'length', method, section_type=section_type)


@feature(shape=(...,))
def bifurcation_partitions(neurite, section_type=NeuriteType.all):
    """Partition at bf points."""
    return _map_sections(
        bf.bifurcation_partition, neurite, Section.ibifurcation_point, section_type=section_type
    )


@feature(shape=(...,))
def sibling_ratios(neurite, method='first', section_type=NeuriteType.all):
    """Sibling ratios at bf points.

    The sibling ratio is the ratio between the diameters of the
    smallest and the largest child. It is a real number between
    0 and 1. Method argument allows one to consider mean diameters
    along the child section instead of diameter of the first point.
    """
    return _map_sections(
        partial(bf.sibling_ratio, method=method),
        neurite,
        Section.ibifurcation_point,
        section_type=section_type,
    )


@feature(shape=(..., 2))
def partition_pairs(neurite, section_type=NeuriteType.all):
    """Partition pairs at bf points.

    Partition pair is defined as the number of bifurcations at the two
    daughters of the bifurcating section
    """
    return _map_sections(
        bf.partition_pair, neurite, Section.ibifurcation_point, section_type=section_type
    )


@feature(shape=(...,))
def diameter_power_relations(neurite, method='first', section_type=NeuriteType.all):
    """Calculate the diameter power relation at a bf point.

    Diameter power relation is defined in https://www.ncbi.nlm.nih.gov/pubmed/18568015

    This quantity gives an indication of how far the branching is from
    the Rall ratio (when =1).
    """
    return _map_sections(
        partial(bf.diameter_power_relation, method=method),
        neurite,
        Section.ibifurcation_point,
        section_type=section_type,
    )


def _radial_distances(neurite, origin, iterator_type, section_type):
    if origin is None:
        origin = neurite.root_node.points[0]

    return _map_sections(
        partial(sf.section_radial_distance, origin=origin),
        neurite=neurite,
        iterator_type=iterator_type,
        section_type=section_type,
    )


@feature(shape=(...,))
def section_radial_distances(neurite, origin=None, section_type=NeuriteType.all):
    """Section radial distances.

    The iterator_type can be used to select only terminal sections (ileaf)
    or only bifurcations (ibifurcation_point).
    """
    return _radial_distances(neurite, origin, Section.ipreorder, section_type)


@feature(shape=(...,))
def section_term_radial_distances(neurite, origin=None, section_type=NeuriteType.all):
    """Get the radial distances of the termination sections."""
    return _radial_distances(neurite, origin, Section.ileaf, section_type)


@feature(shape=())
def max_radial_distance(neurite, origin=None, section_type=NeuriteType.all):
    """Get the maximum radial distances of the termination sections."""
    term_radial_distances = section_term_radial_distances(
        neurite, origin=origin, section_type=section_type
    )
    return max(term_radial_distances) if term_radial_distances else 0.0


@feature(shape=(...,))
def section_bif_radial_distances(neurite, origin=None, section_type=NeuriteType.all):
    """Get the radial distances of the bf sections."""
    return _radial_distances(neurite, origin, Section.ibifurcation_point, section_type)


@feature(shape=(...,))
def terminal_path_lengths(neurite, section_type=NeuriteType.all):
    """Get the path lengths to each terminal point."""
    return section_path_distances(neurite, iterator_type=Section.ileaf, section_type=section_type)


@feature(shape=())
def volume_density(neurite, section_type=NeuriteType.all):
    """Get the volume density.

    The volume density is defined as the ratio of the neurite volume and
    the volume of the neurite's enclosing convex hull

    TODO: convex hull fails on some morphologies, it may be good to instead use
        bounding_box to compute the neurite enclosing volume

    .. note:: Returns `np.nan` if the convex hull computation fails.
    """
    neurite_volume = total_volume(neurite, section_type=section_type)

    def get_points(section):
        return section.points[:, COLS.XYZ].tolist()

    # note: duplicate points included but not affect the convex hull calculation
    points = list(utils.flatten(_map_sections(get_points, neurite, section_type=section_type)))

    hull = convex_hull(points)

    return neurite_volume / hull.volume if hull is not None else np.nan


@feature(shape=(...,))
def section_volumes(neurite, section_type=NeuriteType.all):
    """Section volumes."""
    return _map_sections(sf.section_volume, neurite, section_type=section_type)


@feature(shape=(...,))
def section_areas(neurite, section_type=NeuriteType.all):
    """Section areas."""
    return _map_sections(sf.section_area, neurite, section_type=section_type)


@feature(shape=(...,))
def section_tortuosity(neurite, section_type=NeuriteType.all):
    """Section tortuosities."""
    return _map_sections(sf.section_tortuosity, neurite, section_type=section_type)


@feature(shape=(...,))
def section_end_distances(neurite, section_type=NeuriteType.all):
    """Section end to end distances."""
    return _map_sections(sf.section_end_distance, neurite, section_type=section_type)


@feature(shape=(...,))
def principal_direction_extents(neurite, direction=0, section_type=NeuriteType.all):
    """Principal direction extent of neurites in morphologies.

    Note:
        Principal direction extents are always sorted in descending order. Therefore,
        by default the maximal principal direction extent is returned.
    """
    points = list(iter_points(neurite, section_filter=is_type(section_type)))

    return [morphmath.principal_direction_extent(np.unique(points, axis=0))[direction]]


@feature(shape=(...,))
def section_strahler_orders(neurite, section_type=NeuriteType.all):
    """Inter-segment opening angles in a section."""
    return _map_sections(sf.strahler_order, neurite, section_type=section_type)
