# Copyright (c) 2015, Ecole Polytechnique Federal de Lausanne, Blue Brain Project
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
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''Mathematics functions used to compute morphometrics'''
from math import acos
from itertools import combinations
import numpy as np
from itertools import islice, izip


np.seterr(all='raise')  # raise exceptions for floating point errors.


def point_dist(p1, p2):
    '''compute the euclidian distance between two 3D points

    Args:
        p1, p2: indexable objects with
        indices 0, 1, 2 corresponding to 3D cartesian coordinates.
    Returns:
        The euclidian distance between the points.
    '''
    return np.linalg.norm(np.subtract(p1[0:3], p2[0:3]))


def vector(p1, p2):
    '''compute vector between two 3D points

    Args:
        p1, p2: indexable objects with
        indices 0, 1, 2 corresponding to 3D cartesian coordinates.

    Returns:
        3-vector from p1 - p2
    '''
    return np.subtract(p1[0:3], p2[0:3])


def angle_3points(p0, p1, p2):
    ''' compute the angle in radians between two 3D points

    Calculated as the angle between p0-p1 and p0-p2.

    Args:
        p0, p1, p2:  indexable objects with
        indices 0, 1, 2 corresponding to 3D cartesian coordinates.

    Returns:
        Angle in radians between (p0-p1) and (p0-p2)
    '''
    vec1 = vector(p0, p1)
    vec2 = vector(p0, p2)
    return acos(np.dot(vec1, vec2) /
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
