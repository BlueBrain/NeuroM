'''Mathematics functions used to compute morphometrics'''
from math import acos
from itertools import combinations
import numpy as np


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
