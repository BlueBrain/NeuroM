from nose import tools as nt
from nose.tools import assert_raises
from neurom.core.point import Point
from neurom.analysis.morphmath import point_dist
from neurom.analysis.morphmath import Vector3D
from neurom.analysis.morphmath import vector
from neurom.analysis.morphmath import dot
from neurom.analysis.morphmath import angle_3points
from neurom.analysis.morphmath import norm
from neurom.analysis.morphmath import polygon_diameter
from math import sqrt, sin, cos, pi, fabs

import unittest
from numpy.random import uniform
import numpy as np

def test_vector3D():
    p = Vector3D(2.5, 3.9, 4.6)
    nt.ok_(p.x == 2.5)
    nt.ok_(p.y == 3.9)
    nt.ok_(p.z == 4.6)
    


def test_point_dist():
    p1 = Point(1, 3.0, 4.0, 5.0, 3.0)
    p2 = Point(1, 4.0, 5.0, 6.0, 3.0)
    dist = point_dist(p1,p2)
    nt.ok_(dist==sqrt(3))

def test_vector():
    vec1 = Vector3D(12.0, 3, 6.0)
    vec2 = Vector3D(14.0, 1.0, 6.0)
    vec = vector(vec1,vec2)
    nt.ok_(vec.x==-2.0)
    nt.ok_(vec.y==2.0)
    nt.ok_(vec.z==0.0)
    

def test_dot():
    vec1 = Vector3D(1.0, 0.0, 1.0)
    vec2 = Vector3D(1.0, 1.0, 1.0)
    d = dot(vec1,vec2)
    nt.ok_(d==2.0)

def test_norm():
    vec1 = Vector3D(1.0, 0.0, 1.0)
    d = norm(vec1)
    nt.ok_(d==sqrt(2.0))


def test_angle_3points():
    vec1 = Vector3D(1.0, 0.0, 0.0)
    vec2 = Vector3D(0.0, 1.0, 0.0)
    orig = Vector3D(0.0,0.0,0.0)
    angle=angle_3points(orig, vec1, vec2)
    nt.ok_(angle==pi/2.0)


def test_angle_3points_throws_exception():
    vec1 = Vector3D(1.0, 0.0, 0.0)
    vec2 = Vector3D(0.0, 1.0, 0.0)
    orig = Vector3D(0.0,1.0,0.0)
    assert_raises(ZeroDivisionError, angle_3points,orig,vec1,vec2)


def soma_points(radius=5,number_points=20):
    phi = uniform(0, 2*pi, number_points)
    costheta = uniform(-1,1, number_points)
    theta = np.arccos(costheta)
    x = radius*np.sin(theta)*np.cos(phi)
    y = radius*np.sin(theta)*np.sin(phi)
    z = radius*np.cos(theta)
    surfPoint = [i for i in range(number_points)]
    for i , p in enumerate(zip(x,y,z)):
        surfPoint[i]=Point(1, p[0], p[1], p[2], 0.0)
    return surfPoint
        



def test_polygon_diameter():
    p1 = Point(1, 3.0, 4.0, 5.0, 3.0)
    p2 = Point(1, 3.0, 5.0, 5.0, 3.0)
    p3 = Point(1, 3.0, 6.0, 5.0, 3.0)
    dia= polygon_diameter([p1,p2,p3])
    nt.ok_(dia==2.0)
    surfpoint= soma_points()
    dia1 = polygon_diameter(surfpoint)
    nt.ok_(fabs(dia1-10.0) < 0.1)
    
    
    

    
    

    


    
    
    
