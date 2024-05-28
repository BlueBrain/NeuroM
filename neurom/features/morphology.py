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

"""Morphology features.

Any public function from this namespace can be called via the features mechanism. If calling
directly the function in this namespace can only accept a morphology as its input. If you want to
apply it to a morphology population then you must use the features mechanism e.g. ``features.get``.
The features mechanism does not allow you to apply these features to neurites.

>>> import neurom
>>> from neurom import features
>>> m = neurom.load_morphology("tests/data/swc/Neuron.swc")
>>> result = features.get('soma_surface_area', m)
>>> population = neurom.load_morphologies("tests/data/valid_set")
>>> result = features.get('sholl_crossings', population)

For more details see :ref:`features`.
"""

import warnings
from collections.abc import Iterable
from functools import partial

import numpy as np

import neurom.core.soma
from neurom import morphmath
from neurom.core.dataformat import COLS
from neurom.core.morphology import (
    Morphology,
    iter_neurites,
    iter_points,
    iter_sections,
    iter_segments,
)
from neurom.core.types import NeuriteType
from neurom.core.types import tree_type_checker as is_type
from neurom.exceptions import NeuroMError
from neurom.features import NameSpace, feature
from neurom.features import neurite as nf
from neurom.features import section as sf
from neurom.morphmath import convex_hull
from neurom.utils import flatten, str_to_plane

feature = partial(feature, namespace=NameSpace.NEURON)


def _assert_soma_center(morph):
    if morph.soma.center is None:
        raise NeuroMError(
            f"The morphology named '{morph.name}' has no soma so the feature can not be computed."
        )
    return morph


def _map_neurites(function, morph, neurite_type):
    return list(
        iter_neurites(
            obj=morph,
            mapfun=function,
            filt=is_type(neurite_type),
        )
    )


def _map_neurite_root_nodes(function, morph, neurite_type):
    if neurite_type == NeuriteType.all:
        filt = None
    else:

        def filt(neurite):
            return neurite_type == neurite.type.root_type

    return [function(trunk.root_node) for trunk in iter_neurites(obj=morph, filt=filt)]


def _filter_mode(obj, neurite_type):
    if obj.process_subtrees:
        return {"section_filter": is_type(neurite_type)}
    return {"neurite_filter": is_type(neurite_type)}


def _get_sections(morph, neurite_type):
    return list(iter_sections(morph, **_filter_mode(morph, neurite_type)))


def _get_segments(morph, neurite_type):
    return list(iter_segments(morph, **_filter_mode(morph, neurite_type)))


def _get_points(morph, neurite_type):
    return list(iter_points(morph, **_filter_mode(morph, neurite_type)))


@feature(shape=())
def soma_volume(morph):
    """Get the volume of a morphology's soma."""
    return neurom.core.soma.get_volume(morph.soma)


@feature(shape=())
def soma_surface_area(morph):
    """Get the surface area of a morphology's soma.

    Note:
        The surface area is calculated by assuming the soma is spherical.
    """
    return neurom.core.soma.get_area(morph.soma)


@feature(shape=())
def soma_radius(morph):
    """Get the radius of a morphology's soma."""
    return neurom.core.soma.get_radius(morph.soma)


@feature(shape=())
def max_radial_distance(morph, origin=None, neurite_type=NeuriteType.all):
    """Get the maximum radial distances of the termination sections."""
    origin = morph.soma.center if origin is None else origin

    term_radial_distances = _map_neurites(
        partial(nf.max_radial_distance, origin=origin), morph, neurite_type
    )
    return max(term_radial_distances) if term_radial_distances else 0.0


@feature(shape=(...,))
def section_radial_distances(morph, origin=None, neurite_type=NeuriteType.all):
    """Section radial distances."""
    origin = morph.soma.center if origin is None else origin

    return list(
        flatten(
            _map_neurites(
                partial(nf.section_radial_distances, origin=origin),
                morph=morph,
                neurite_type=neurite_type,
            )
        )
    )


@feature(shape=(...,))
def section_term_radial_distances(morph, origin=None, neurite_type=NeuriteType.all):
    """Get the radial distances of the termination sections."""
    origin = morph.soma.center if origin is None else origin

    return list(
        flatten(
            _map_neurites(
                partial(nf.section_term_radial_distances, origin=origin),
                morph=morph,
                neurite_type=neurite_type,
            )
        )
    )


@feature(shape=(...,))
def section_bif_radial_distances(morph, origin=None, neurite_type=NeuriteType.all):
    """Get the radial distances of the bifurcation sections."""
    origin = morph.soma.center if origin is None else origin

    return list(
        flatten(
            _map_neurites(
                partial(nf.section_bif_radial_distances, origin=origin),
                morph=morph,
                neurite_type=neurite_type,
            )
        )
    )


@feature(shape=(...,))
def segment_radial_distances(morph, origin=None, neurite_type=NeuriteType.all):
    """Ger the radial distances of the segments."""
    origin = morph.soma.center if origin is None else origin

    return list(
        flatten(
            _map_neurites(
                partial(nf.segment_radial_distances, origin=origin),
                morph=morph,
                neurite_type=neurite_type,
            )
        )
    )


@feature(shape=(...,))
def number_of_sections_per_neurite(morph, neurite_type=NeuriteType.all):
    """List of numbers of sections per neurite."""
    return _map_neurites(nf.number_of_sections, morph, neurite_type)


@feature(shape=(...,))
def total_length_per_neurite(morph, neurite_type=NeuriteType.all):
    """Neurite lengths."""
    return _map_neurites(nf.total_length, morph, neurite_type)


@feature(shape=(...,))
def total_area_per_neurite(morph, neurite_type=NeuriteType.all):
    """Neurite areas."""
    return _map_neurites(nf.total_area, morph, neurite_type)


@feature(shape=(...,))
def total_volume_per_neurite(morph, neurite_type=NeuriteType.all):
    """Neurite volumes."""
    return _map_neurites(nf.total_volume, morph, neurite_type)


@feature(shape=(...,))
def trunk_origin_azimuths(morph, neurite_type=NeuriteType.all):
    """Get a list of all the trunk origin azimuths of a morph.

    The azimuth is defined as Angle between x-axis and the vector
    defined by (initial tree point - soma center) on the x-z plane.

    The range of the azimuth angle [-pi, pi] radians
    """
    _assert_soma_center(morph)

    def azimuth(root_node):
        """Azimuth of a neurite trunk."""
        return morphmath.azimuth_from_vector(
            morphmath.vector(root_node.points[0], morph.soma.center)
        )

    return _map_neurite_root_nodes(azimuth, morph, neurite_type)


@feature(shape=(...,))
def trunk_origin_elevations(morph, neurite_type=NeuriteType.all):
    """Get a list of all the trunk origin elevations of a morph.

    The elevation is defined as the angle between x-axis and the
    vector defined by (initial tree point - soma center)
    on the x-y half-plane.

    The range of the elevation angle [-pi/2, pi/2] radians
    """
    _assert_soma_center(morph)

    def elevation(root_node):
        """Elevation of a section."""
        return morphmath.elevation_from_vector(
            morphmath.vector(root_node.points[0], morph.soma.center)
        )

    return _map_neurite_root_nodes(elevation, morph, neurite_type)


@feature(shape=(...,))
def trunk_vectors(morph, neurite_type=NeuriteType.all):
    """Calculate the vectors between all the trunks of the morphology and the soma center."""
    _assert_soma_center(morph)

    def vector_from_soma_to_root(root_node):
        return morphmath.vector(root_node.points[0], morph.soma.center)

    return _map_neurite_root_nodes(vector_from_soma_to_root, morph, neurite_type)


@feature(shape=(...,))
def trunk_angles(
    morph,
    neurite_type=NeuriteType.all,
    coords_only="xy",
    sort_along="xy",
    consecutive_only=True,
):
    """Calculate the angles between all the trunks of the morph.

    By default, the angles are defined on the x-y plane and the trees are sorted from the y axis
    and anticlock-wise.

    Args:
        morph: The morphology to process.
        neurite_type: Only the neurites of this type are considered.
        coords_only: Consider only the coordinates listed in this argument (should be a combination
            of 'x', 'y' and 'z').
        sort_along: Sort angles according to the given plane (should be 'xy', 'xz' or 'yz') before
            computing the angles between the trunks.
        consecutive_only: Compute only the angles between consecutive trunks (the default order of
            neurite trunks is the same as the one used by
            :func:`neurom.core.morphology.iter_neurites` but this order can be changed using the
            `sort_along` parameter).

    Returns:
        list[[float]] or list[float]:
            The angles between each trunk and all the others. If ``consecutive_only`` is ``True``,
            only the angle with the next trunk is returned for each trunk.
    """
    vectors = np.array(trunk_vectors(morph, neurite_type=neurite_type))
    # In order to avoid the failure of the process in case the neurite_type does not exist
    if len(vectors) == 0:
        return []

    if sort_along:
        # Sorting angles according to the given plane
        sort_coords = str_to_plane(sort_along)
        order = np.argsort(
            np.fromiter(
                (
                    morphmath.angle_between_projections(i / np.linalg.norm(i), [0, 1])
                    for i in vectors[:, sort_coords]
                ),
                dtype=float,
            )
        )
        vectors = vectors[order]

    # Select coordinates to consider
    if coords_only:
        coords = str_to_plane(coords_only)
        vectors = vectors[:, coords]

    # Compute angles between each trunk and the next ones
    n_vectors = len(vectors)
    cycling_vectors = np.vstack([vectors, vectors])
    angles = [
        (
            num_i,
            [
                morphmath.angle_between_vectors(i, j)
                for j in cycling_vectors[num_i : num_i + n_vectors]
            ],
        )
        for num_i, i in enumerate(vectors)
    ]

    if consecutive_only:
        angles = [i[1][-1] for i in angles if i[1]]
    else:
        angles = [i[1] for i in angles]

    return angles


@feature(shape=(...,))
def trunk_angles_inter_types(
    morph,
    source_neurite_type=NeuriteType.apical_dendrite,
    target_neurite_type=NeuriteType.basal_dendrite,
    closest_component=None,
):
    """Calculate the angles between the trunks of the morph of a source type to target type.

    Args:
        morph: The morphology to process.
        source_neurite_type: Only the neurites of this type are considered as sources.
        target_neurite_type: Only the neurites of this type are considered as targets.
        closest_component:
            If ``closest_component`` is not ``None``, only one element is returned for each neurite
            of source type:

            * if set to 0, the one with the lowest absolute 3d angle is returned.
            * if set to 1, the one with the lowest absolute elevation angle is returned.
            * if set to 2, the one with the lowest absolute azimuth angle is returned.

    Returns:
        list[list[float]] or list[float]:
            If ``closest_component`` is ``None``, a list of 3 elements is returned for each couple
            of neurites:

            * the absolute 3d angle between the two vectors.
            * the elevation angle (or polar angle) between the two vectors.
            * the azimuth angle between the two vectors.

            If ``closest_component`` is not ``None``, only one of these values is returned for each
            couple.
    """
    source_vectors = trunk_vectors(morph, neurite_type=source_neurite_type)
    target_vectors = trunk_vectors(morph, neurite_type=target_neurite_type)

    # In order to avoid the failure of the process in case the neurite_type does not exist
    if len(source_vectors) == 0 or len(target_vectors) == 0:
        return []

    angles = np.empty((len(source_vectors), len(target_vectors), 3), dtype=float)

    for i, source in enumerate(source_vectors):
        for j, target in enumerate(target_vectors):
            angles[i, j, 0] = morphmath.angle_between_vectors(source, target)
            angles[i, j, [1, 2]] = morphmath.spherical_from_vector(
                target
            ) - morphmath.spherical_from_vector(source)

    # Ensure elevation differences are in [-pi, pi]
    angles[:, :, 1] = morphmath.angles_to_pi_interval(angles[:, :, 1])

    # Ensure azimuth differences are in [-2pi, 2pi]
    angles[:, :, 2] = morphmath.angles_to_pi_interval(angles[:, :, 2], scale=2.0)

    if closest_component is not None:
        angles = angles[
            np.arange(len(angles)), np.argmin(np.abs(angles[:, :, closest_component]), axis=1)
        ][:, np.newaxis, :]

    return angles.tolist()


@feature(shape=(...,))
def trunk_angles_from_vector(
    morph,
    neurite_type=NeuriteType.all,
    vector=None,
):
    """Calculate the angles between the trunks of the morph of a given type and a given vector.

    Args:
        morph: The morphology to process.
        neurite_type: Only the neurites of this type are considered.
        vector: The reference vector. If ``None``, the reference vector is set to ``(0, 1, 0)``.

    Returns:
        list[list[float]]:
            For each neurite, an array with 3 elements is returned:

            * the absolute 3d angle between the two vectors.
            * the elevation angle (or polar angle) between the two vectors.
            * the azimuth angle between the two vectors.
    """
    if vector is None:
        vector = (0, 1, 0)

    vectors = np.array(trunk_vectors(morph, neurite_type=neurite_type))

    # In order to avoid the failure of the process in case the neurite_type does not exist
    if len(vectors) == 0:
        return []

    angles = np.empty((len(vectors), 3), dtype=float)
    for i, i_vec in enumerate(vectors):
        angles[i, 0] = morphmath.angle_between_vectors(vector, i_vec)
        angles[i, (1, 2)] = morphmath.spherical_from_vector(
            i_vec
        ) - morphmath.spherical_from_vector(vector)

    # Ensure elevation difference are in [-pi, pi]
    angles[:, 1] = morphmath.angles_to_pi_interval(angles[:, 1])

    # Ensure azimuth difference are in [-2pi, 2pi]
    angles[:, 2] = morphmath.angles_to_pi_interval(angles[:, 2], scale=2)

    return angles.tolist()


@feature(shape=(...,))
def trunk_origin_radii(
    morph,
    neurite_type=NeuriteType.all,
    min_length_filter=None,
    max_length_filter=None,
):
    """Radii of the trunk sections of neurites in a morph.

    .. warning::
        If ``min_length_filter`` and / or ``max_length_filter`` is given, the points are filtered
        and the mean radii of the remaining points is returned.
        Note that if the ``min_length_filter`` is greater than the path distance of the last point
        of the first section, the radius of this last point is returned.

    Args:
        morph: The morphology to process.
        neurite_type: Only the neurites of this type are considered.
        min_length_filter: The min length from which the neurite points are considered.
        max_length_filter: The max length from which the neurite points are considered.

    Returns:
        list[float]:
            * if ``min_length_filter`` and ``max_length_filter`` are ``None``, the radii of the
              first point of each neurite are returned.
            * else the mean radius of the points between the given ``min_length_filter`` and
              ``max_length_filter`` are returned.
    """
    if min_length_filter is not None and min_length_filter <= 0:
        raise NeuroMError(
            "In 'trunk_origin_radii': the 'min_length_filter' value must be strictly greater "
            "than 0."
        )

    if max_length_filter is not None and max_length_filter <= 0:
        raise NeuroMError(
            "In 'trunk_origin_radii': the 'max_length_filter' value must be strictly greater "
            "than 0."
        )

    if (
        min_length_filter is not None
        and max_length_filter is not None
        and min_length_filter >= max_length_filter
    ):
        raise NeuroMError(
            "In 'trunk_origin_radii': the 'min_length_filter' value must be strictly less than the "
            "'max_length_filter' value."
        )

    def trunk_first_radius(root_node):
        return root_node.points[0][COLS.R]

    def trunk_mean_radius(root_node):
        points = root_node.points

        interval_lengths = morphmath.interval_lengths(points)
        path_lengths = np.insert(np.cumsum(interval_lengths), 0, 0)
        valid_pts = np.ones(len(path_lengths), dtype=bool)

        if min_length_filter is not None:
            valid_pts = valid_pts & (path_lengths >= min_length_filter)
            if not valid_pts.any():
                warnings.warn(
                    "In 'trunk_origin_radii': the 'min_length_filter' value is greater than the "
                    "path distance of the last point of the last section so the radius of this "
                    "point is returned."
                )
                return points[-1, COLS.R]

        if max_length_filter is not None:
            valid_max = path_lengths <= max_length_filter
            valid_pts = valid_pts & valid_max
            if not valid_pts.any():
                warnings.warn(
                    "In 'trunk_origin_radii': the 'min_length_filter' and 'max_length_filter' "
                    "values excluded all the points of the section so the radius of the first "
                    "point after the 'min_length_filter' path distance is returned."
                )
                # pylint: disable=invalid-unary-operand-type
                return points[~valid_max, COLS.R][0]

        return points[valid_pts, COLS.R].mean()

    function = (
        trunk_first_radius
        if max_length_filter is None and min_length_filter is None
        else trunk_mean_radius
    )

    return _map_neurite_root_nodes(function, morph, neurite_type)


@feature(shape=(...,))
def trunk_section_lengths(morph, neurite_type=NeuriteType.all):
    """List of lengths of trunk sections of neurites in a morph."""
    return _map_neurite_root_nodes(sf.section_length, morph, neurite_type)


@feature(shape=())
def number_of_neurites(morph, neurite_type=NeuriteType.all):
    """Number of neurites in a morph."""
    return len(_map_neurites(lambda x, section_type: 1, morph, neurite_type))


@feature(shape=(...,))
def neurite_volume_density(morph, neurite_type=NeuriteType.all):
    """Get volume density per neurite."""
    return _map_neurites(nf.volume_density, morph, neurite_type)


@feature(shape=(...,))
def sholl_crossings(morph, neurite_type=NeuriteType.all, center=None, radii=None):
    """Calculate crossings of neurites.

    Args:
        morph(Morphology|list): morphology or a list of neurites
        neurite_type(NeuriteType): Type of neurite to use. By default ``NeuriteType.all`` is used.
        center(Point): center point, if None then soma center is taken
        radii: iterable of floats for which crossings will be counted,
            if None then soma radius is taken

    Returns:
        Array of same length as radii, with a count of the number of crossings
        for the respective radius

    This function can also be used with a list of sections, as follow::

        secs = (sec for sec in nm.iter_sections(morph) if complex_filter(sec))
        sholl = nm.features.neuritefunc.sholl_crossings(secs,
                                                        center=morph.soma.center,
                                                        radii=np.arange(0, 1000, 100))
    """

    def count_crossings(section, radius):
        """Used to count crossings of segments in neurite with radius."""
        r2 = radius**2
        count = 0
        for start, end in iter_segments(section):
            start_dist2, end_dist2 = (
                morphmath.point_dist2(center, start),
                morphmath.point_dist2(center, end),
            )

            if start_dist2 <= r2 <= end_dist2 or end_dist2 <= r2 <= start_dist2:
                count += 1

        return count

    if center is None or radii is None:
        assert isinstance(morph, Morphology) and morph.soma, (
            '`sholl_crossings` input error. If `center` or `radii` is not set then `morph` is '
            'expected to be an instance of Morphology and have a soma.'
        )
        if center is None:
            _assert_soma_center(morph)
            center = morph.soma.center
        if radii is None:
            radii = [morph.soma.radius]

    if isinstance(morph, Iterable):
        sections = filter(is_type(neurite_type), morph)
    else:
        sections = _get_sections(morph, neurite_type)

    counts_per_radius = [0 for _ in range(len(radii))]

    for section in sections:
        for i, radius in enumerate(radii):
            counts_per_radius[i] += count_crossings(section, radius)

    return counts_per_radius


@feature(shape=(...,))
def sholl_frequency(morph, neurite_type=NeuriteType.all, step_size=10, bins=None):
    """Perform Sholl frequency calculations on a morph.

    Args:
        morph(Morphology): a morphology
        neurite_type(NeuriteType): which neurites to operate on
        step_size(float): step size between Sholl radii
        bins: iterable of floats defining custom binning to use for the Sholl radii.
            If None, it uses intervals of step_size between min and max radii of ``morphologies``.

    Note:
        Given a morphology, the soma center is used for the concentric circles,
        which range from the soma radii, and the maximum radial distance
        in steps of `step_size`. Each segment of the morphology is tested, so a neurite that
        bends back on itself, and crosses the same Sholl radius will get counted as
        having crossed multiple times.

        If a `neurite_type` is specified and there are no trees corresponding to it, an empty
        list will be returned.
    """
    _assert_soma_center(morph)

    if bins is None:
        min_soma_edge = morph.soma.radius

        sections = _get_sections(morph, neurite_type)

        max_radius_per_section = [
            np.max(np.linalg.norm(section.points[:, COLS.XYZ] - morph.soma.center, axis=1))
            for section in sections
        ]

        if not max_radius_per_section:
            return []

        bins = np.arange(min_soma_edge, min_soma_edge + max(max_radius_per_section), step_size)

    return sholl_crossings(morph, neurite_type, morph.soma.center, bins)


def _extent_along_axis(morph, axis, neurite_type):
    """Returns the total extent of the morpholog neurites.

    The morphology is filtered by neurite type and the extent is calculated
    along the coordinate axis direction (e.g. COLS.X).
    """
    points = _get_points(morph, neurite_type)

    if not points:
        return 0.0

    return abs(np.ptp(np.asarray(points)[:, axis]))


@feature(shape=())
def total_width(morph, neurite_type=NeuriteType.all):
    """Extent of morphology along axis x."""
    return _extent_along_axis(morph, COLS.X, neurite_type)


@feature(shape=())
def total_height(morph, neurite_type=NeuriteType.all):
    """Extent of morphology along axis y."""
    return _extent_along_axis(morph, COLS.Y, neurite_type)


@feature(shape=())
def total_depth(morph, neurite_type=NeuriteType.all):
    """Extent of morphology along axis z."""
    return _extent_along_axis(morph, COLS.Z, neurite_type)


@feature(shape=())
def volume_density(morph, neurite_type=NeuriteType.all):
    """Get the volume density.

    The volume density is defined as the ratio of the neurite volume and
    the volume of the morphology's enclosing convex hull

    .. note:: Returns `np.nan` if the convex hull computation fails or there are not points
              available due to neurite type filtering.
    """
    points = _get_points(morph, neurite_type)

    if not points:
        return np.nan

    morph_hull = convex_hull(points)

    if morph_hull is None:
        return np.nan

    total_volume = sum(total_volume_per_neurite(morph, neurite_type=neurite_type))

    return total_volume / morph_hull.volume


def _unique_projected_points(morph, projection_plane, neurite_type):
    key = "".join(sorted(projection_plane.lower()))

    try:
        axes = {"xy": COLS.XY, "xz": COLS.XZ, "yz": COLS.YZ}[key]

    except KeyError as e:
        raise NeuroMError(
            f"Invalid 'projection_plane' argument {projection_plane}. "
            f"Please select 'xy', 'xz', or 'yz'."
        ) from e

    points = _get_points(morph, neurite_type)

    if len(points) == 0:
        return np.empty(shape=(0, 3), dtype=np.float32)

    return np.unique(np.vstack(points), axis=0)[:, axes]


@feature(shape=())
def aspect_ratio(morph, neurite_type=NeuriteType.all, projection_plane="xy"):
    """Calculates the min/max ratio of the principal direction extents along the plane.

    Args:
        morph: Morphology object.
        neurite_type: The neurite type to use. By default all neurite types are used.
        projection_plane: Projection plane to use for the calculation. One of ('xy', 'xz', 'yz').

    Returns:
        The aspect ratio feature of the morphology points.
    """
    projected_points = _unique_projected_points(morph, projection_plane, neurite_type)
    return np.nan if len(projected_points) == 0 else morphmath.aspect_ratio(projected_points)


@feature(shape=())
def circularity(morph, neurite_type=NeuriteType.all, projection_plane="xy"):
    """Calculates the circularity of the morphology points along the plane.

    The circularity is defined as the 4 * pi * area of the convex hull over its
    perimeter.

    Args:
        morph: Morphology object.
        neurite_type: The neurite type to use. By default all neurite types are used.
        projection_plane: Projection plane to use for the calculation. One of
            ('xy', 'xz', 'yz').

    Returns:
        The circularity of the morphology points.
    """
    projected_points = _unique_projected_points(morph, projection_plane, neurite_type)
    return np.nan if len(projected_points) == 0 else morphmath.circularity(projected_points)


@feature(shape=())
def shape_factor(morph, neurite_type=NeuriteType.all, projection_plane="xy"):
    """Calculates the shape factor of the morphology points along the plane.

    The shape factor is defined as the ratio of the convex hull area over max squared
    pairwise distance of the morphology points.

    Args:
        morph: Morphology object.
        neurite_type: The neurite type to use. By default all neurite types are used.
        projection_plane: Projection plane to use for the calculation. One of
            ('xy', 'xz', 'yz').

    Returns:
        The shape factor of the morphology points.
    """
    projected_points = _unique_projected_points(morph, projection_plane, neurite_type)
    return np.nan if len(projected_points) == 0 else morphmath.shape_factor(projected_points)


@feature(shape=())
def length_fraction_above_soma(morph, neurite_type=NeuriteType.all, up="Y"):
    """Returns the length fraction of the segments that have their midpoints higher than the soma.

    Args:
        morph: Morphology object.
        neurite_type: The neurite type to use. By default all neurite types are used.
        up: The axis along which the computation is performed. One of ('X', 'Y', 'Z').

    Returns:
        The fraction of neurite length that lies on the right of the soma along the given axis.
    """
    _assert_soma_center(morph)
    axis = up.upper()

    if axis not in {"X", "Y", "Z"}:
        raise NeuroMError(f"Unknown axis {axis}. Please choose 'X', 'Y', or 'Z'.")

    col = getattr(COLS, axis)

    segments = _get_segments(morph, neurite_type)

    if not segments:
        return np.nan

    # (Segment 1, Segment 2) x (X, Y, Z, R) X N
    segments = np.dstack(segments)

    # shape N x 3
    seg_begs = segments[0, COLS.XYZ, :].T
    seg_ends = segments[1, COLS.XYZ, :].T

    lengths = np.linalg.norm(seg_begs - seg_ends, axis=1)

    midpoints = 0.5 * (seg_begs + seg_ends)
    selection = midpoints[:, col] > morph.soma.center[col]

    return lengths[selection].sum() / lengths.sum()
