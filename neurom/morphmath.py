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

"""Mathematical and geometrical functions used to compute morphometrics."""
import logging
import math
from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

try:
    # The QhulError was moved in scipy >= 1.8 so if the import fails the old location is imported
    from scipy.spatial import QhullError
except ImportError:  # pragma: no cover
    from scipy.spatial.qhull import QhullError

from neurom.core.dataformat import COLS

L = logging.getLogger(__name__)


def vector(p1, p2):
    """Compute vector between two 3D points.

    Args:
        p1, p2: indexable objects with indices 0, 1, 2 corresponding to 3D cartesian coordinates.

    Returns:
        3-vector from p1 - p2
    """
    return np.subtract(p1[COLS.XYZ], p2[COLS.XYZ])


def linear_interpolate(p1, p2, fraction):
    """Returns the point p satisfying: p1 + fraction * (p2 - p1)."""
    return np.array(
        (
            p1[0] + fraction * (p2[0] - p1[0]),
            p1[1] + fraction * (p2[1] - p1[1]),
            p1[2] + fraction * (p2[2] - p1[2]),
        )
    )


def interpolate_radius(r1, r2, fraction):
    """Interpolate the radius between two values.

    Calculate the radius that corresponds to a point P that lies at a fraction of the length
    of a cut cone P1P2 where P1, P2 are the centers of the circles that bound the shape with radii
    r1 and r2 respectively.

    Args:
        r1: float
            Radius of the first node of the segment.
        r2: float
            Radius of the second node of the segment
        fraction: float
            The fraction at which the interpolated radius is calculated.

    Returns: float
        The interpolated radius.

    Note: The fraction is assumed from point P1, not from point P2.
    """

    def f(a, b, c):
        """Returns the length of the interpolated radius calculated using similar triangles."""
        return a + c * (b - a)

    return f(r2, r1, 1.0 - fraction) if r1 > r2 else f(r1, r2, fraction)


def interval_lengths(points, prepend_zero=False):
    """Returns the list of distances between consecutive points.

    Args:
        points: a list of np.array of 3D points
        prepend_zero (bool): if True, the returned array will start with a zero
    """
    intervals = np.linalg.norm(np.diff(np.asarray(points)[:, COLS.XYZ], axis=0), axis=1)
    if prepend_zero:
        return np.insert(intervals, 0, 0)
    return intervals


def path_fraction_id_offset(points, fraction, relative_offset=False):
    """Find segment by fractional offset.

    Find the segment which corresponds to the fraction
    of the path length along the piecewise linear curve which
    is constructed from the set of points.

    Args:
        points: an iterable of indexable objects with indices
            0, 1, 2 correspoding to 3D cartesian coordinates
        fraction: path length fraction (0.0 <= fraction <= 1.0)
        relative_offset: return absolute or relative segment distance

    Returns:
        (segment ID, segment offset) pair.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Invalid fraction: %.3f" % fraction)
    lengths = interval_lengths(points)
    cum_lengths = np.cumsum(lengths)
    offset = cum_lengths[-1] * fraction
    seg_id = np.argmin(cum_lengths < offset)
    if seg_id > 0:
        offset -= cum_lengths[seg_id - 1]
    if relative_offset:
        offset /= lengths[seg_id]
    return seg_id, offset


def path_fraction_point(points, fraction):
    """Find coordinates by fractional offset.

    Computes the point which corresponds to the fraction
    of the path length along the piecewise linear curve which
    is constructed from the set of points.

    Args:
        points: an iterable of indexable objects with indices
            0, 1, 2 correspoding to 3D cartesian coordinates
        fraction: path length fraction (0 <= fraction <= 1)

    Returns:
        The 3D coordinates of the aforementioned point
    """
    seg_id, offset = path_fraction_id_offset(points, fraction, relative_offset=True)
    return linear_interpolate(points[seg_id], points[seg_id + 1], offset)


def scalar_projection(v1, v2):
    """Compute the scalar projection of v1 upon v2.

    Args:
        v1, v2: iterable indices 0, 1, 2 corresponding to cartesian coordinates

    Returns:
        3-vector of the projection of point p onto the direction of v
    """
    return np.dot(v1, v2) / np.linalg.norm(v2)


def vector_projection(v1, v2):
    """Compute the vector projection of v1 upon v2.

    Args:
        v1, v2: iterable indices 0, 1, 2 corresponding to cartesian coordinates

    Returns:
        3-vector of the projection of point p onto the direction of v
    """
    return scalar_projection(v1, v2) * v2 / np.linalg.norm(v2)


def dist_point_line(p, l1, l2):
    """Compute the orthogonal distance between a line and a point.

    The line is that which passes through the points l1 and l2.

    Args:
        p: iterable
            indices 0, 1, 2 correspond to cartesian coordinates
        l1: iterable
            indices 0, 1, 2 correspond to cartesian coordinates
        l2: iterable
            indices 0, 1, 2 correspond to cartesian coordinates
    """
    cross_prod = np.cross(l2 - l1, p - l1)
    return np.linalg.norm(cross_prod) / np.linalg.norm(l2 - l1)


def point_dist2(p1, p2):
    """Compute the square of the euclidian distance between two 3D points.

    Args:
        p1, p2: indexable objects with indices 0, 1, 2 corresponding to 3D cartesian coordinates.

    Returns:
        The square of the euclidian distance between the points.
    """
    v = vector(p1, p2)
    return np.dot(v, v)


def point_dist(p1, p2):
    """Compute the euclidian distance between two 3D points.

    Args:
        p1, p2: indexable objects with indices 0, 1, 2 corresponding to 3D cartesian coordinates.

    Returns:
        The euclidian distance between the points.
    """
    return np.sqrt(point_dist2(p1, p2))


def angle_3points(p0, p1, p2):
    """Compute the angle in radians between three 3D points.

    Calculated as the angle between p1-p0 and p2-p0.

    Args:
        p0, p1, p2:  indexable objects with
            indices 0, 1, 2 corresponding to 3D cartesian coordinates.

    Returns:
        Angle in radians between (p1-p0) and (p2-p0).
        0.0 if p0==p1 or p0==p2.
    """
    vec1 = vector(p1, p0)
    vec2 = vector(p2, p0)
    return math.atan2(np.linalg.norm(np.cross(vec1, vec2)), np.dot(vec1, vec2))


def angle_between_vectors(p1, p2):
    """Computes the angle in radians between vectors 'p1' and 'p2'.

    Normalizes the input vectors and computes the relative angle
    between them.
    """
    if np.equal(p1, p2).all():
        return 0.0
    v1 = p1 / np.linalg.norm(p1)
    v2 = p2 / np.linalg.norm(p2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def angle_between_projections(p1, p2):
    """Angle between the projections p1 and p2 (2d vectors)."""
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return ang1 - ang2


def elevation_from_vector(vec):
    """Return the elevation of a vector."""
    norm_vector = np.linalg.norm(vec)

    if norm_vector >= np.finfo(type(norm_vector)).eps:
        return np.arcsin(np.clip(vec[COLS.Y] / norm_vector, -1.0, 1.0))
    raise ValueError("Norm of vector between soma center and section is almost zero.")


def azimuth_from_vector(vec):
    """Return the azimuth of a vector."""
    return np.arctan2(vec[COLS.Z], vec[COLS.X])


def spherical_from_vector(vec):
    """Return the spherical coordinates of a vector: elevation and azimuth.

    .. note::

        * the elevation is the angle between the vector and its projection on the XZ plane.
        * the azimuth is the angle between the X axis and the projection of the vector on the XZ
          plane.

    .. warning:: This frame is not the usual spherical frame (see :ref:`spherical_coordinates`).
    """
    # Azimuth is in [-pi, pi]
    azimuth = azimuth_from_vector(vec)

    # Elevation is in [-pi/2, pi/2]
    elevation = elevation_from_vector(vec)

    return np.array([elevation, azimuth])


def vector_from_spherical(elevation, azimuth, radius=1.0):
    """Return a vector from the frame center to the point in given direction and given radius.

    .. warning:: The frame is not the usual spherical frame (see :ref:`spherical_coordinates`).
    """
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.sin(elevation)
    z = np.cos(elevation) * np.sin(azimuth)

    return radius * np.array([x, y, z])


def angles_to_pi_interval(angles, scale=1.0):
    """Convert any angles into the ]-scale * pi, scale * pi] interval."""
    mod_angle = np.fmod(angles, 2.0 * scale * np.pi)
    mod_angle = np.where(mod_angle <= -scale * np.pi, mod_angle + 2 * scale * np.pi, mod_angle)
    mod_angle = np.where(mod_angle > scale * np.pi, mod_angle - 2 * scale * np.pi, mod_angle)
    return mod_angle


def polygon_diameter(points):
    """Compute the maximun euclidian distance between any two points in a list of points."""
    return max(point_dist(p0, p1) for (p0, p1) in combinations(points, 2))


def average_points_dist(p0, p_list):
    """Computes the average distance between a list of points and a given point p0."""
    return np.mean(list(point_dist(p0, p1) for p1 in p_list))


def path_distance(points):
    """Compute the path distance from given set of points."""
    return interval_lengths(points).sum()


def segment_length(seg):
    """Return the length of a segment.

    Returns: Euclidian distance between centres of points in seg
    """
    return point_dist(seg[0], seg[1])


def segment_length2(seg):
    """Return the square of the length of a segment.

    Returns: Square of Euclidian distance between centres of points in seg
    """
    return point_dist2(seg[0], seg[1])


def segment_radius(seg):
    """Return the mean radius of a segment.

    Returns: arithmetic mean of the radii of the points in seg
    """
    return (seg[0][COLS.R] + seg[1][COLS.R]) / 2.0


def segment_x_coordinate(seg):
    """Return the mean x coordinate of a segment.

    Returns: arithmetic mean of the x coordinates of the points in seg
    """
    return (seg[0][COLS.X] + seg[1][COLS.X]) / 2.0


def segment_y_coordinate(seg):
    """Return the mean y coordinate of a segment.

    Returns: arithmetic mean of the y coordinates of the points in seg
    """
    return (seg[0][COLS.Y] + seg[1][COLS.Y]) / 2.0


def segment_z_coordinate(seg):
    """Return the mean z coordinate of a segment.

    Returns: arithmetic mean of the z coordinates of the points in seg
    """
    return (seg[0][COLS.Z] + seg[1][COLS.Z]) / 2.0


def segment_radial_dist(seg, pos):
    """Return the radial distance of a tree segment to a given point.

    The radial distance is the euclidian distance between the mid-point of
    the segment and the point in question.

    Arguments:
        seg: tree segment
        pos: origin to which distances are measured.
            It must have at lease 3 components. The first 3 components are (x, y, z).
    """
    return point_dist(pos, np.divide(np.add(seg[0], seg[1]), 2.0))


def segment_area(seg):
    """Compute the surface area of a segment.

    Approximated as a conical frustum. Does not include the surface area
    of the bounding circles.
    """
    r0 = seg[0][COLS.R]
    r1 = seg[1][COLS.R]
    h2 = point_dist2(seg[0], seg[1])
    return math.pi * (r0 + r1) * math.sqrt((r0 - r1) ** 2 + h2)


def segment_volume(seg):
    """Compute the volume of a segment.

    Approximated as a conical frustum.
    """
    r0 = seg[0][COLS.R]
    r1 = seg[1][COLS.R]
    h = point_dist(seg[0], seg[1])
    return math.pi * h * ((r0 * r0) + (r0 * r1) + (r1 * r1)) / 3.0


def taper_rate(p0, p1):
    """Compute the taper rate between points p0 and p1.

    Args:
        p0, p1: iterables with first 4 components containing (x, y, z, r)

    Returns:
        The taper rate, defined as the absolute value of the difference in
        the diameters of p0 and p1 divided by the euclidian distance
        between them.
    """
    return 2 * abs(p0[COLS.R] - p1[COLS.R]) / point_dist(p0, p1)


def segment_taper_rate(seg):
    """Compute the taper rate of a segment.

    Returns:
        The taper rate, defined as the absolute value of the difference in
        the diameters of the segment's two points divided by the euclidian
        distance between them.
    """
    return taper_rate(seg[0], seg[1])


def pca(points):
    """Estimate the principal components of the covariance on the given point cloud.

    Args:
        points: A numpy array of points of the form ((x1,y1,z1), (x2, y2, z2)...)

    Returns:
        Eigenvalues and respective eigenvectors
    """
    return np.linalg.eig(np.cov(points.transpose()))


def sphere_area(r):
    """Compute the area of a sphere with radius r."""
    return 4.0 * math.pi * r**2


# Useful alias for path_distance
section_length = path_distance


def principal_direction_extent(points):
    """Calculate the extent of a set of 3D points.

    The extent is defined as the maximum distance between the projections on the principal
    directions of the covariance matrix of the points.

    Args:
        points: a 2D numpy array of points with 2 or 3 columns for (x, y, z)

    Returns:
        the extents for each of the eigenvectors of the cov matrix

    Note:
        Direction extents are ordered from largest to smallest.
    """
    # pca can be biased by duplicate points
    points = np.unique(points, axis=0)

    # center the points around 0.0
    points -= np.mean(points, axis=0)

    # principal components
    _, eigenvectors = pca(points)

    # for each eigenvector calculate the scalar projection of the points on it (n_points, n_eigv)
    scalar_projections = points.dot(eigenvectors)

    # range of the projections (abs(max - min)) along each column (eigenvector)
    extents = np.ptp(scalar_projections, axis=0)

    descending_order = np.argsort(extents)[::-1]

    return extents[descending_order]


def convex_hull(points):
    """Get the convex hull from an array of points.

    Returns:
        scipy.spatial.ConvexHull object if successful, otherwise None
    """
    if len(points) == 0:
        L.exception("Failure to compute convex hull because there are no points")
        return None

    try:
        return ConvexHull(points)
    except QhullError:
        L.exception("Failure to compute convex hull because of geometrical degeneracy.")
        return None


def aspect_ratio(points):
    """Computes the min/max ratio of the principal direction extents."""
    extents = principal_direction_extent(points)
    return float(extents.min() / extents.max())


def circularity(points):
    """Computes circularity as 4 * pi * area / perimeter^2.

    Note: For 2D points, ConvexHull.volume corresponds to its area and ConvexHull.area
        to its perimeter.
    """
    hull = convex_hull(points)
    return 4.0 * np.pi * hull.volume / hull.area**2


def shape_factor(points):
    """Computes area over max pairwise distance squared.

    Defined in doi: 10.1109/ICoAC44903.2018.8939083

    Note: For 2D points, ConvexHull.volume corresponds to its area.
    """
    hull = convex_hull(points)
    hull_points = points[hull.vertices]

    max_pairwise_distance = np.max(cdist(hull_points, hull_points))

    return hull.volume / max_pairwise_distance**2
