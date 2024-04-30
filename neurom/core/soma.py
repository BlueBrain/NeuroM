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

"""Soma classes and functions."""
import math
import warnings

import morphio
import numpy as np
from morphio import SomaError, SomaType

from neurom import morphmath
from neurom.core.dataformat import COLS


class Soma:
    """Base class for a soma.

    Holds a list of raw data rows corresponding to soma points
    and provides iterator access to them.
    """

    def __init__(self, morphio_soma):
        """Constructor.

        Args:
            morphio_soma (morphio.Soma): instance of soma of MorphIO class
        """
        self._morphio_soma = morphio_soma

    def to_morphio(self):
        """Return morphio instance."""
        return self._morphio_soma

    @property
    def center(self):
        """Obtain the center from the first stored point."""
        return get_center(self)

    @property
    def radius(self):
        """Return radius of soma."""
        # this radius is used only for `volume` method, please avoid using it for anything else.
        return get_radius(self)

    def iter(self):
        """Iterator to soma contents."""
        return iter(self.points)

    @property
    def points(self):
        """Get the set of (x, y, z, r) points this soma."""
        return np.concatenate(
            (self._morphio_soma.points, self._morphio_soma.diameters[:, np.newaxis] / 2.0), axis=1
        )

    @property
    def area(self):
        """Calculate soma area."""
        return get_area(self)

    @property
    def volume(self):
        """Calculate soma volume."""
        return get_volume(self)

    def overlaps(self, points, exclude_boundary=False):
        """Check that the given points are located inside the soma."""
        return check_overlaps(self, points, exclude_boundary=exclude_boundary)


class SomaSinglePoint(Soma):
    """Type A: 1point soma.

    Represented by a single point.
    """

    def __str__(self):
        """Return a string representation."""
        return 'SomaSinglePoint(%s) <center: %s, radius: %s>' % (
            repr(self.points),
            self.center,
            self.radius,
        )


class SomaCylinders(Soma):
    """Soma composed of cylinders (like in SWC).

    points describe the locations of the cylinder start/end points, with
    their respective radii, much like how neurites are described:

        ex::

                             /)
                            / o)
                    ______ /  )
                (|)      ) / /
                ( o )    o )/
                (|)_____ )

      Here we have a 'side-view', with each 'o' representing a point, and the
      radius is the height of a '|' character, and the ')' try and show the
      curvature of the cylinger

    Note: when, as in the case above, the cylinder center points don't lie
    in a line, then the overlap between cylinders isn't taken into account for
    the area calculation
    """

    def __str__(self):
        """Return a string representation."""
        return 'SomaCylinders(%s) <center: %s, virtual radius: %s>' % (
            repr(self.points),
            self.center,
            self.radius,
        )


class SomaNeuromorphoThreePointCylinders(SomaCylinders):
    """NeuroMorpho compatible soma.

    Reference:
        http://neuromorpho.org/SomaFormat.html

        Quote (Tue Feb 28 14:26:56 CET 2017):
        The first point constitutes the center of the soma, with coordinates (xs, ys,
        zs) corresponding to the average of all the points in the single contour. An
        equivalent radius (rs) is computed as the average distance of each point of the
        single contour from this center. The first point is assigned parent ID -1 as is
        standard for the root in the SWC format. The second and third soma points, as
        well as all starting points (roots) of dendritic and axonal arbors have this
        first point as the parent (parent ID 1). The x and z coordinates as well as the
        radius of both the second and third soma points are the same as that of the
        first point. The y coordinates of the second and third points are shifted by -rs
        and +rs, respectively (Figure 2). The surface area of this soma cylinder equals
        the surface area of a sphere of radius rs.
    """

    def __init__(self, morphio_soma):
        """Initialize a SomaNeuromorphoThreePointCylinders object."""
        super().__init__(morphio_soma)
        # X    Y     Z   R    P
        # xs ys      zs rs   -1
        # xs (ys-rs) zs rs    1
        # xs (ys+rs) zs rs    1

        r1, r2, r3 = 0.5 * morphio_soma.diameters

        # make sure the above invariant holds
        assert np.isclose(r1, r2) and np.isclose(r1, r3), 'All radii must be the same'

        if r1 < 1e-5:
            warnings.warn('Zero radius for {}'.format(self))

    def __str__(self):
        """Return a string representation."""
        return 'SomaNeuromorphoThreePointCylinders(%s) <center: %s, radius: %s>' % (
            repr(self.points),
            self.center,
            self.radius,
        )


class SomaSimpleContour(Soma):
    """Type C: multiple points soma.

    Represented by a contour.

    The equivalent radius as the average distance to the center.

    Note: This doesn't currently check to see if the contour is in a plane. Also
    the radii of the points are not taken into account.
    """

    def __str__(self):
        """Return a string representation."""
        return 'SomaSimpleContour(%s) <center: %s, radius: %s>' % (
            repr(self.points),
            self.center,
            self.radius,
        )


def _morphio_soma(soma):
    """Return morphio soma."""
    if isinstance(soma, Soma):
        return soma.to_morphio()
    if isinstance(soma, (morphio.Soma, morphio.mut.Soma)):
        return soma
    raise TypeError(f"Unknown soma type {type(soma)}")


def _dispatch_soma_functions(soma, dispatch_mapping, **kwargs):
    morphio_soma = _morphio_soma(soma)
    soma_algo = dispatch_mapping[morphio_soma.type]
    return soma_algo(morphio_soma, **kwargs)


def get_center(soma):
    """Calculate soma center."""
    dispatch_mapping = {
        SomaType.SOMA_UNDEFINED: _first_point_or_none,
        SomaType.SOMA_SINGLE_POINT: _first_point,
        SomaType.SOMA_CYLINDERS: _first_point,
        SomaType.SOMA_NEUROMORPHO_THREE_POINT_CYLINDERS: _first_point,
        SomaType.SOMA_SIMPLE_CONTOUR: _centroid,
    }
    return _dispatch_soma_functions(soma, dispatch_mapping)


def _first_point(morphio_soma):
    """Return the first point."""
    return morphio_soma.points[0]


def _first_point_or_none(morphio_soma):
    """Return first point if there is at least one point or None otherwise."""
    points = morphio_soma.points
    return points[0] if len(points) > 0 else None


def _centroid(morphio_soma):
    """Return the centroid of the soma points."""
    return np.mean(morphio_soma.points, axis=0)


def get_radius(soma):
    """Calculate soma radius."""
    dispatch_mapping = {
        SomaType.SOMA_UNDEFINED: lambda _: 0,
        SomaType.SOMA_SINGLE_POINT: _soma_single_point_radius,
        SomaType.SOMA_CYLINDERS: _soma_cylinders_radius,
        SomaType.SOMA_NEUROMORPHO_THREE_POINT_CYLINDERS: _soma_three_point_cylinders_radius,
        SomaType.SOMA_SIMPLE_CONTOUR: _soma_simple_contour_radius,
    }
    return _dispatch_soma_functions(soma, dispatch_mapping)


def _soma_single_point_radius(morphio_soma):
    """Return first radius."""
    return 0.5 * morphio_soma.diameters[0]


def _soma_cylinders_radius(morphio_soma):
    """Calculate radius calculated from the cylinder area."""
    points = np.concatenate(
        (morphio_soma.points, 0.5 * morphio_soma.diameters[:, np.newaxis]),
        axis=1,
    )
    area = sum(morphmath.segment_area((p0, p1)) for p0, p1 in zip(points, points[1:]))
    return math.sqrt(area / (4.0 * math.pi))


def _soma_three_point_cylinders_radius(morphio_soma):
    """Calculate three-point-cylinder radius."""
    return math.sqrt(get_area(morphio_soma) / (4.0 * math.pi))


def _soma_simple_contour_radius(morphio_soma):
    """Calculate average contour distance from center of soma."""
    return morphmath.average_points_dist(get_center(morphio_soma), morphio_soma.points)


def get_area(soma):
    """Calculate soma area."""
    dispatch_mapping = {
        SomaType.SOMA_UNDEFINED: _soma_undefined_area,
        SomaType.SOMA_SINGLE_POINT: _soma_single_point_area,
        SomaType.SOMA_CYLINDERS: _soma_cylinders_area,
        SomaType.SOMA_NEUROMORPHO_THREE_POINT_CYLINDERS: _soma_three_point_cylinders_area,
        SomaType.SOMA_SIMPLE_CONTOUR: _soma_single_point_area,
    }
    return _dispatch_soma_functions(soma, dispatch_mapping)


def _soma_single_point_area(morphio_soma):
    """Calculate soma area as a sphere."""
    return 4.0 * math.pi * get_radius(morphio_soma) ** 2


def _soma_undefined_area(morphio_soma):
    """Calculate soma as a sphere."""
    warnings.warn('Approximating soma area by a sphere. {}'.format(morphio_soma))
    return _soma_single_point_area(morphio_soma)


def _soma_cylinders_area(morphio_soma):
    """Calculate soma area as a sum of cylinder surfaces withouts caps."""
    points = np.concatenate(
        (
            morphio_soma.points,
            0.5 * morphio_soma.diameters[:, np.newaxis],
        ),
        axis=1,
    )
    return sum(morphmath.segment_area((p0, p1)) for p0, p1 in zip(points, points[1:]))


def _soma_three_point_cylinders_area(morphio_soma):
    """Calculate soma area as a cylinder comprised of three points with the same radius."""
    r = 0.5 * morphio_soma.diameters[0]
    h = morphmath.point_dist(morphio_soma.points[1], morphio_soma.points[2])
    return 2.0 * math.pi * r * h  # ignores the 'end-caps' of the cylinder


def get_volume(soma):
    """Calculate soma volume."""
    dispatch_mapping = {
        SomaType.SOMA_UNDEFINED: _soma_undefined_volume,
        SomaType.SOMA_SINGLE_POINT: _soma_single_point_volume,
        SomaType.SOMA_CYLINDERS: _soma_cylinders_volume,
        SomaType.SOMA_NEUROMORPHO_THREE_POINT_CYLINDERS: _soma_three_point_cylinders_volume,
        SomaType.SOMA_SIMPLE_CONTOUR: _soma_undefined_volume,
    }
    return _dispatch_soma_functions(soma, dispatch_mapping)


def _soma_single_point_volume(morphio_soma):
    """Calculate soma volume as a sphere."""
    return 4.0 / 3 * math.pi * get_radius(morphio_soma) ** 3


def _soma_undefined_volume(morphio_soma):
    """Calculate soma volume as a sphere."""
    warnings.warn('Approximating soma volume by a sphere. {}'.format(morphio_soma))
    return _soma_single_point_volume(morphio_soma)


def _soma_cylinders_volume(morphio_soma):
    """Calculate soma volume as a sum of cylinder volumes."""
    points = np.concatenate(
        (
            morphio_soma.points,
            0.5 * morphio_soma.diameters[:, np.newaxis],
        ),
        axis=1,
    )
    return sum(morphmath.segment_volume((p0, p1)) for p0, p1 in zip(points, points[1:]))


def _soma_three_point_cylinders_volume(morphio_soma):
    """Calculate soma volume as a cylinder of three points and same radius."""
    return 2.0 * math.pi * get_radius(morphio_soma) ** 3


def check_overlaps(soma, points, exclude_boundary=False):
    """Check if soma overlaps with points."""
    dispatch_mapping = {
        SomaType.SOMA_UNDEFINED: _soma_undefined_overlaps,
        SomaType.SOMA_SINGLE_POINT: _soma_undefined_overlaps,
        SomaType.SOMA_CYLINDERS: _soma_cylinders_overlaps,
        SomaType.SOMA_NEUROMORPHO_THREE_POINT_CYLINDERS: _soma_cylinders_overlaps,
        SomaType.SOMA_SIMPLE_CONTOUR: _soma_simple_contour_overlaps,
    }
    return _dispatch_soma_functions(
        soma,
        dispatch_mapping,
        points=points,
        exclude_boundary=exclude_boundary,
    )


def _soma_undefined_overlaps(morphio_soma, points, exclude_boundary):
    """Check if points overlap with soma approximated as a sphere."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))

    center = get_center(morphio_soma)
    radius = get_radius(morphio_soma)

    if exclude_boundary:
        return np.linalg.norm(points - center, axis=1) < radius

    return np.linalg.norm(points - center, axis=1) <= radius


def _soma_cylinders_overlaps(morphio_soma, points, exclude_boundary):
    """Check if points overlap with soma approximated as a collection of cylinders."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))

    soma_points = np.concatenate(
        (
            morphio_soma.points,
            0.5 * morphio_soma.diameters[:, np.newaxis],
        ),
        axis=1,
    )

    mask = np.ones(len(points)).astype(bool)
    for p1, p2 in zip(soma_points[:-1], soma_points[1:]):
        vec = p2[COLS.XYZ] - p1[COLS.XYZ]
        vec_norm = np.linalg.norm(vec)
        dot = (points[mask] - p1[COLS.XYZ]).dot(vec) / vec_norm

        cross = np.linalg.norm(np.cross(vec, points[mask]), axis=1) / vec_norm
        dot_clipped = np.clip(dot / vec_norm, a_min=0, a_max=1)
        radii = p1[COLS.R] * (1 - dot_clipped) + p2[COLS.R] * dot_clipped

        if exclude_boundary:
            in_cylinder = (dot > 0) & (dot < vec_norm) & (cross < radii)
        else:
            in_cylinder = (dot >= 0) & (dot <= vec_norm) & (cross <= radii)
        mask[np.where(mask)] = ~in_cylinder
        if not mask.any():
            break

    return ~mask


def _soma_simple_contour_overlaps(morphio_soma, points, exclude_boundary):
    """Check if points overlap with soma approximated as a contour."""
    # pylint: disable=too-many-locals
    soma_points = np.concatenate(
        (
            morphio_soma.points,
            0.5 * morphio_soma.diameters[:, np.newaxis],
        ),
        axis=1,
    )
    center = get_center(morphio_soma)

    points = np.atleast_2d(np.asarray(points, dtype=np.float64))

    # Convert points to angles from the center
    relative_pts = points - center
    pt_angles = np.arctan2(relative_pts[:, COLS.Y], relative_pts[:, COLS.X])

    # Convert soma points to angles from the center
    relative_soma_pts = soma_points[:, COLS.XYZ] - center
    soma_angles = np.arctan2(relative_soma_pts[:, COLS.Y], relative_soma_pts[:, COLS.X])

    # Order the soma points by ascending angles
    soma_angle_order = np.argsort(soma_angles)
    ordered_soma_angles = soma_angles[soma_angle_order]
    ordered_relative_soma_pts = relative_soma_pts[soma_angle_order]

    # Find the two soma points which form the segment crossed by the one from the center
    # to the point
    angles = np.atleast_2d(pt_angles).T - ordered_soma_angles
    closest_indices = np.argmin(np.abs(angles), axis=1)
    neighbors = np.ones_like(closest_indices)
    neighbors[angles[np.arange(len(closest_indices)), closest_indices] < 0] = -1
    signs = (neighbors == 1) * 2.0 - 1.0
    neighbors[(closest_indices >= len(relative_soma_pts) - 1) & (neighbors == 1)] = (
        -len(relative_soma_pts) + 1
    )

    # Compute the cross product and multiply by neighbors to get the same result as if all
    # vectors were clockwise
    cross_z = (
        np.cross(
            (
                ordered_relative_soma_pts[closest_indices + neighbors]
                - ordered_relative_soma_pts[closest_indices]
            ),
            relative_pts - ordered_relative_soma_pts[closest_indices],
        )[:, COLS.Z]
        * signs
    )

    if exclude_boundary:
        interior_side = cross_z > 0
    else:
        interior_side = cross_z >= 0

    return interior_side


def make_soma(morphio_soma):
    """Make a soma object from a MorphIO soma.

    Args:
        morphio_soma(morphio.Soma): soma instance of MorphIO
    """
    soma_builders = {
        SomaType.SOMA_SINGLE_POINT: SomaSinglePoint,
        SomaType.SOMA_CYLINDERS: SomaCylinders,
        SomaType.SOMA_NEUROMORPHO_THREE_POINT_CYLINDERS: SomaNeuromorphoThreePointCylinders,
        SomaType.SOMA_SIMPLE_CONTOUR: SomaSimpleContour,
        SomaType.SOMA_UNDEFINED: Soma,
    }

    builder = soma_builders.get(morphio_soma.type)
    if builder is None:
        raise SomaError(f'No NeuroM constructor for MorphIO soma type: {morphio_soma.type}')
    return builder(morphio_soma)
