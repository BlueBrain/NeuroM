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
from itertools import islice, izip
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
    return np.linalg.norm(vector(p1, p2))


def angle_3points(p0, p1, p2):
    ''' compute the angle in radians between three 3D points

    Calculated as the angle between p1-p0 and p2-p0.

    Args:
        p0, p1, p2:  indexable objects with
        indices 0, 1, 2 corresponding to 3D cartesian coordinates.

    Returns:
        Angle in radians between (p1-p0) and (p2-p0).
        nan if p0==p1 or p0==p2.
    '''
    vec1 = vector(p1, p0)
    vec2 = vector(p2, p0)
    return math.acos(np.dot(vec1, vec2) /
                     (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


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
    return sum(point_dist(p[0], p[1]) for p in izip(points, islice(points, 1, None)))


def segment_length(seg):
    '''Return the length of a segment.

    Returns: Euclidian distance between centres of points in seg
    '''
    return point_dist(seg[0], seg[1])


def segment_radius(seg):
    '''Return the mean radius of a segment

    Returns: arithmetic mean of the radii of the points in seg
    '''
    return (seg[0][COLS.R] + seg[1][COLS.R]) / 2.


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


# Useful alias for path_distance
section_length = path_distance
