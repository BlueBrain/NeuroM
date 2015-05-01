from nose import tools as nt
from neurom.core.point import Point
from neurom.analysis.morphmath import point_dist
from neurom.analysis.morphmath import vector
from neurom.analysis.morphmath import angle_3points
from neurom.analysis.morphmath import polygon_diameter
from neurom.analysis.morphmath import average_points_dist
from math import sqrt, pi, fabs

from numpy.random import uniform
import numpy as np


np.random.seed(0)


def test_point_dist():
    p1 = Point(3.0, 4.0, 5.0, 3.0, 1)
    p2 = Point(4.0, 5.0, 6.0, 3.0, 1)
    dist = point_dist(p1,p2)
    nt.ok_(dist==sqrt(3))


def test_vector():
    vec1 = (12.0, 3, 6.0)
    vec2 = (14.0, 1.0, 6.0)
    vec = vector(vec1,vec2)
    nt.ok_(np.all(vec ==(-2.0, 2.0, 0.0)))


def test_angle_3points():
    vec1 = (1.0, 0.0, 0.0)
    vec2 = (0.0, 1.0, 0.0)
    orig = (0.0,0.0,0.0)
    angle=angle_3points(orig, vec1, vec2)
    nt.ok_(angle==pi/2.0)


@nt.raises(ArithmeticError)
def test_angle_3points_throws_exception():
    vec1 = (1.0, 0.0, 0.0)
    vec2 = (0.0, 1.0, 0.0)
    orig = (0.0,1.0,0.0)
    angle_3points(orig,vec1,vec2)


def soma_points(radius=5,number_points=20):
    phi = uniform(0, 2*pi, number_points)
    costheta = uniform(-1,1, number_points)
    theta = np.arccos(costheta)
    x = radius*np.sin(theta)*np.cos(phi)
    y = radius*np.sin(theta)*np.sin(phi)
    z = radius*np.cos(theta)

    return [Point(i, j, k, 0.0, 1) for (i, j, k) in zip(x, y, z)]


def test_polygon_diameter():
    p1 = Point(3.0, 4.0, 5.0, 3.0, 1)
    p2 = Point(3.0, 5.0, 5.0, 3.0, 1)
    p3 = Point(3.0, 6.0, 5.0, 3.0, 1)
    dia= polygon_diameter([p1,p2,p3])
    nt.ok_(dia==2.0)
    surfpoint= soma_points()
    dia1 = polygon_diameter(surfpoint)
    nt.ok_(fabs(dia1-10.0) < 0.1)


def test_average_points_dist():
    p0 = Point(0.0, 0.0, 0.0, 3.0, 1)
    p1 = Point(0.0, 0.0, 1.0, 3.0, 1)
    p2 = Point(0.0, 0.0, 1.0, 3.0, 1)
    av_dist = average_points_dist(p0, [p1,p1])
    nt.ok_(av_dist == 1.0)
