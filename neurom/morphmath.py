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

'''Mathematical and geometrical functions used to compute morphometrics'''
import math
from itertools import combinations
import numpy as np
from neurom.core.dataformat import COLS


def vector(p1, p2):
    '''compute vector between two 3D points

    Args:
        p1, p2: indexable objects with
        indices 0, 1, 2 corresponding to 3D cartesian coordinates.

    Returns:
        3-vector from p1 - p2
    '''
    return np.subtract(p1[0:3], p2[0:3])


def linear_interpolate(p1, p2, fraction):
    '''Returns the point p satisfying |p1p| = fraction * |p1p2|'''
    return np.array((p1[0] + fraction * (p2[0] - p1[0]),
                     p1[1] + fraction * (p2[1] - p1[1]),
                     p1[2] + fraction * (p2[2] - p1[2])))


def interpolate_radius(r1, r2, fraction):
    '''Calculate the radius that corresponds to a point P that lies at a fraction of the length
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
    '''
    def f(a, b, c):
        ''' Returns the length of the interpolated radius calculated
        using similar triangles.
        '''
        return a + c * (b - a)
    return f(r2, r1, 1. - fraction) if r1 > r2 else f(r1, r2, fraction)


def path_fraction_point(points, fraction):
    '''Computes the point which corresponds to the fraction
    of the path length along the piecewise linear curve which
    is constructed from the set of points.

    Args:
        points: an iterable of indexable objects with indices
        0, 1, 2 correspoding to 3D cartesian coordinates

    Returns:
        The 3D coordinates of the aforementioned point
    '''
    def path_until_threshold(points, fraction_path_length):
        ''' Calculates the cummulative path length of the
        line segments until the threshold frac_length is met. It
        returns the two points between which lies the point that
        corresponds to the fraction and the cummulative length.
        '''
        n = 0
        cummulative_length = point_dist(points[0], points[1])
        # stop if the cummulative path length becomes
        # greater or equal to the desired one or
        # if all points are used up
        while cummulative_length < fraction_path_length and n <= len(points) - 1:
            n += 1
            cummulative_length += point_dist(points[n], points[n + 1])
        return points[n], points[n + 1], cummulative_length

    frac_length = fraction * path_distance(points)
    p0, p1, cumm_length = path_until_threshold(points, frac_length)
    fraction = 1. - (cumm_length - frac_length) / point_dist(p0, p1)
    return linear_interpolate(p0, p1, fraction)


def scalar_projection(v1, v2):
    '''compute the scalar projection of v1 upon v2

    Args:
        v1, v2: iterable
        indices 0, 1, 2 corresponding to cartesian coordinates

    Returns:
        3-vector of the projection of point p onto the direction of v
    '''
    return np.dot(v1, v2) / np.linalg.norm(v2)


def vector_projection(v1, v2):
    '''compute the vector projection of v1 upon v2

    Args:
        v1, v2: iterable
        indices 0, 1, 2 corresponding to cartesian coordinates

    Returns:
        3-vector of the projection of point p onto the direction of v
    '''
    return scalar_projection(v1, v2) * v2 / np.linalg.norm(v2)


def dist_point_line(p, l1, l2):
    '''compute the orthogonal distance between from the line that goes through
    the points l1, l2 and the point p

    Args:
        p, l1, l2 : iterable
        point
        indices 0, 1, 2 corresponding to cartesian coordinates
    '''
    cross_prod = np.cross(l2 - l1, p - l1)
    return np.linalg.norm(cross_prod) / np.linalg.norm(l2 - l1)


def point_dist2(p1, p2):
    '''compute the square of the euclidian distance between two 3D points

    Args:
        p1, p2: indexable objects with
        indices 0, 1, 2 corresponding to 3D cartesian coordinates.
    Returns:
        The square of the euclidian distance between the points.
    '''
    v = vector(p1, p2)
    return np.dot(v, v)


def point_dist(p1, p2):
    '''compute the euclidian distance between two 3D points

    Args:
        p1, p2: indexable objects with
        indices 0, 1, 2 corresponding to 3D cartesian coordinates.
    Returns:
        The euclidian distance between the points.
    '''
    return np.sqrt(point_dist2(p1, p2))


def angle_3points(p0, p1, p2):
    ''' compute the angle in radians between three 3D points

    Calculated as the angle between p1-p0 and p2-p0.

    Args:
        p0, p1, p2:  indexable objects with
        indices 0, 1, 2 corresponding to 3D cartesian coordinates.

    Returns:
        Angle in radians between (p1-p0) and (p2-p0).
        0.0 if p0==p1 or p0==p2.
    '''
    vec1 = vector(p1, p0)
    vec2 = vector(p2, p0)
    return math.atan2(np.linalg.norm(np.cross(vec1, vec2)), np.dot(vec1, vec2))


def polygon_diameter(points):
    ''' Compute the maximun euclidian distance between any two points
    in a list of points
    '''
    return max(point_dist(p0, p1) for (p0, p1) in combinations(points, 2))


def average_points_dist(p0, p_list):
    """
    Computes the average distance between a list of points
    and a given point p0.
    """
    return np.mean(list(point_dist(p0, p1) for p1 in p_list))


def path_distance(points):
    """
    Compute the path distance from given set of points
    """
    vecs = np.diff(points, axis=0)[:, :3]
    d2 = [np.dot(p, p) for p in vecs]
    return np.sum(np.sqrt(d2))


def segment_length(seg):
    '''Return the length of a segment.

    Returns: Euclidian distance between centres of points in seg
    '''
    return point_dist(seg[0], seg[1])


def segment_length2(seg):
    '''Return the square of the length of a segment.

    Returns: Square of Euclidian distance between centres of points in seg
    '''
    return point_dist2(seg[0], seg[1])


def segment_radius(seg):
    '''Return the mean radius of a segment

    Returns: arithmetic mean of the radii of the points in seg
    '''
    return (seg[0][COLS.R] + seg[1][COLS.R]) / 2.


def segment_x_coordinate(seg):
    '''Return the mean x coordinate of a segment

    Returns: arithmetic mean of the x coordinates of the points in seg
    '''
    return (seg[0][COLS.X] + seg[1][COLS.X]) / 2.


def segment_y_coordinate(seg):
    '''Return the mean y coordinate of a segment

    Returns: arithmetic mean of the y coordinates of the points in seg
    '''
    return (seg[0][COLS.Y] + seg[1][COLS.Y]) / 2.


def segment_z_coordinate(seg):
    '''Return the mean z coordinate of a segment

    Returns: arithmetic mean of the z coordinates of the points in seg
    '''
    return (seg[0][COLS.Z] + seg[1][COLS.Z]) / 2.


def segment_radial_dist(seg, pos):
    '''Return the radial distance of a tree segment to a given point

    The radial distance is the euclidian distance between the mid-point of
    the segment and the point in question.

    Parameters:
        seg: tree segment

        pos: origin to which disrances are measured. It must have at lease 3
        components. The first 3 components are (x, y, z).
    '''
    return point_dist(pos, np.divide(np.add(seg[0], seg[1]), 2.0))


def segment_area(seg):
    '''Compute the surface area of a segment.

    Approximated as a conical frustum. Does not include the surface area
    of the bounding circles.
    '''
    r0 = seg[0][COLS.R]
    r1 = seg[1][COLS.R]
    h2 = point_dist2(seg[0], seg[1])
    return math.pi * (r0 + r1) * math.sqrt((r0 - r1) ** 2 + h2)


def segment_volume(seg):
    '''Compute the volume of a segment.

    Approximated as a conical frustum.
    '''
    r0 = seg[0][COLS.R]
    r1 = seg[1][COLS.R]
    h = point_dist(seg[0], seg[1])
    return math.pi * h * ((r0 * r0) + (r0 * r1) + (r1 * r1)) / 3.0


def taper_rate(p0, p1):
    '''Compute the taper rate between points p0 and p1

    Args:
        p0, p1: iterables with first 4 components containing (x, y, z, r)

    Returns:
        The taper rate, defined as the absolute value of the difference in
        the diameters of p0 and p1 divided by the euclidian distance
        between them.
    '''
    return 2 * abs(p0[COLS.R] - p1[COLS.R]) / point_dist(p0, p1)


def segment_taper_rate(seg):
    '''Compute the taper rate of a segment

    Returns:
        The taper rate, defined as the absolute value of the difference in
        the diameters of the segment's two points divided by the euclidian
        distance between them.
    '''
    return taper_rate(seg[0], seg[1])


def pca(points):
    '''
    Estimate the principal components of the covariance on the given point cloud

    Input
        A numpy array of points of the form ((x1,y1,z1), (x2, y2, z2)...)

    Ouptut
        Eigenvalues and respective eigenvectors
    '''
    return np.linalg.eig(np.cov(points.transpose()))


def sphere_area(r):
    ''' Compute the area of a sphere with radius r
    '''
    return 4. * math.pi * r ** 2

# Useful alias for path_distance
section_length = path_distance


def principal_direction_extent(points):
    '''Calculate the extent of a set of 3D points.

   The extent is defined as the maximum distance between
   the projections on the principal directions of the covariance matrix
   of the points.

   Parameter:
       points : a 2D numpy array of points

   Returns:
       extents : the extents for each of the eigenvectors of the cov matrix
       eigs : eigenvalues of the covariance matrix
       eigv : respective eigenvectors of the covariance matrix
    '''
    # center the points around 0.0
    points -= np.mean(points, axis=0)

    # principal components
    _, eigv = pca(points)

    extent = np.zeros(3)

    for i in xrange(eigv.shape[1]):
        # orthogonal projection onto the direction of the v component
        scalar_projs = np.sort(np.array([np.dot(p, eigv[:, i]) for p in points]))
        extent[i] = scalar_projs[-1]

        if scalar_projs[0] < 0.:
            extent -= scalar_projs[0]

    return extent
