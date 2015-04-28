'''Mathematics functions used to compute morphometrics'''

from math import sqrt
from math import acos
from collections import namedtuple
import itertools

Vector3D = namedtuple('Vector3D', ('x', 'y', 'z'))


def point_dist(point1, point2):
    '''compute the distance between center of two Point'''
    x = point1.x - point2.x
    y = point1.y - point2.y
    z = point1.z - point2.z
    return sqrt(x * x + y * y + z * z)


def vector(point1, point2):
    '''compute vecteur between center of two Point '''
    p = Vector3D(point1.x - point2.x, point1.y - point2.y, point1.z - point2.z)
    return p


def norm(v3d):
    ''' compute norm of vector represent by given 3d point'''
    return sqrt(v3d.x * v3d.x + v3d.y * v3d.y + v3d.z * v3d.z)


def dot(pt1, pt2):
    ''' compute dot product between two 3d point '''
    return pt1.x * pt2.x + pt1.y * pt2.y + pt1.z * pt2.z


def angle_3points(parent, point1, point2):
    ''' compute angle in radian of two Point'''
    vec1 = vector(parent, point1)
    vec2 = vector(parent, point2)
    nvec1 = norm(vec1)
    nvec2 = norm(vec2)
    val = acos(dot(vec1, vec2) / (nvec1 * nvec2))
    return val


def polygon_diameter(listPoints):
    ''' Compute the maximun distance between two points of list of Point'''
    d = max(point_dist(p0, p1) for (p0, p1) in itertools.combinations(listPoints, 2))
    return d
