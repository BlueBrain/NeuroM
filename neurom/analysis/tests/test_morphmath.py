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

from nose import tools as nt
from neurom.core.point import Point
from neurom.analysis.morphmath import point_dist
from neurom.analysis.morphmath import point_dist2
from neurom.analysis.morphmath import vector
from neurom.analysis.morphmath import angle_3points
from neurom.analysis.morphmath import polygon_diameter
from neurom.analysis.morphmath import average_points_dist
from neurom.analysis.morphmath import path_distance
from neurom.analysis.morphmath import segment_length
from neurom.analysis.morphmath import segment_radius
from neurom.analysis.morphmath import segment_volume
from neurom.analysis.morphmath import segment_area
from neurom.analysis.morphmath import segment_radial_dist
from neurom.analysis.morphmath import taper_rate
from neurom.analysis.morphmath import segment_taper_rate
from neurom.analysis.morphmath import pca
from math import sqrt, pi, fabs

from numpy.random import uniform
import numpy as np
import math


np.random.seed(0)


def test_vector():
    vec1 = (12.0, 3, 6.0)
    vec2 = (14.0, 1.0, 6.0)
    vec = vector(vec1,vec2)
    nt.ok_(np.all(vec ==(-2.0, 2.0, 0.0)))


def test_point_dist2():
    p1 = Point(3.0, 4.0, 5.0, 3.0, 1)
    p2 = Point(4.0, 5.0, 6.0, 3.0, 1)
    dist = point_dist2(p1, p2)
    nt.ok_(dist==3)


def test_point_dist():
    p1 = Point(3.0, 4.0, 5.0, 3.0, 1)
    p2 = Point(4.0, 5.0, 6.0, 3.0, 1)
    dist = point_dist(p1,p2)
    nt.ok_(dist==sqrt(3))


def test_angle_3points_half_pi():
    orig = (0.0, 0.0, 0.0)
    vec1 = (1.0, 0.0, 0.0)
    vec2 = (0.0, 2.0, 0.0)
    nt.eq_(angle_3points(orig, vec1, vec2), pi / 2.0)

    vec2 = (0.0, 0.0, 3.0)
    nt.eq_(angle_3points(orig, vec1, vec2), pi / 2.0)

    vec2 = (0.0, 0.0, -3.0)
    nt.eq_(angle_3points(orig, vec1, vec2), pi / 2.0)

    vec1 = (0.0, 4.0, 0.0)
    nt.eq_(angle_3points(orig, vec1, vec2), pi / 2.0)


def test_angle_3points_quarter_pi():
    orig = (0.0, 0.0, 0.0)
    vec1 = (1.0, 0.0, 0.0)
    vec2 = (2.0, 2.0, 0.0)
    nt.assert_equal(angle_3points(orig, vec1, vec2), pi / 4.0)

    vec2 = (3.0, 3.0, 0.0)
    nt.assert_equal(angle_3points(orig, vec1, vec2), pi / 4.0)

    vec2 = (3.0, -3.0, 0.0)
    nt.assert_equal(angle_3points(orig, vec1, vec2), pi / 4.0)

    vec2 = (3.0, 0.0, 3.0)
    nt.assert_equal(angle_3points(orig, vec1, vec2), pi / 4.0)

    vec2 = (3.0, 0.0, -3.0)
    nt.assert_equal(angle_3points(orig, vec1, vec2), pi / 4.0)


def test_angle_3points_three_quarter_pi():
    orig = (0.0, 0.0, 0.0)
    vec1 = (1.0, 0.0, 0.0)
    vec2 = (-2.0, 2.0, 0.0)
    nt.assert_equal(angle_3points(orig, vec1, vec2), 3 * pi / 4.0)

    vec2 = (-3.0, 3.0, 0.0)
    nt.assert_equal(angle_3points(orig, vec1, vec2), 3* pi / 4.0)

    vec2 = (-3.0, -3.0, 0.0)
    nt.assert_equal(angle_3points(orig, vec1, vec2), 3 * pi / 4.0)

    vec2 = (-3.0, 0.0, 3.0)
    nt.assert_equal(angle_3points(orig, vec1, vec2), 3 * pi / 4.0)

    vec2 = (-3.0, 0.0, -3.0)
    nt.assert_equal(angle_3points(orig, vec1, vec2), 3 * pi / 4.0)


def test_angle_3points_equal_points_returns_zero():
    orig = (0.0, 1.0, 0.0)
    vec1 = (1.0, 0.0, 0.0)
    vec2 = (0.0, 1.0, 0.0)
    a = angle_3points(orig, vec1, vec2)
    nt.assert_equal(a, 0.0)


def test_angle_3points_opposing_returns_pi():
    orig = (0.0, 0.0, 0.0)
    vec1 = (1.0, 1.0, 1.0)
    vec2 = (-2.0, -2.0, -2.0)
    angle=angle_3points(orig, vec1, vec2)
    nt.assert_equal(angle, pi)


def test_angle_3points_collinear_returns_zero():
    orig = (0.0, 0.0, 0.0)
    vec1 = (1.0, 1.0, 1.0)
    vec2 = (2.0, 2.0, 2.0)
    angle=angle_3points(orig, vec1, vec2)
    nt.assert_equal(angle, 0.0)


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
    p2 = Point(0.0, 1.0, 0.0, 3.0, 1)
    p3 = Point(1.0, 0.0, 0.0, 3.0, 1)
    av_dist = average_points_dist(p0, [p1, p2, p3])
    nt.ok_(av_dist == 1.0)

def test_path_distance():
    p1 = Point(3.0, 4.0, 5.0, 3.0, 1)
    p2 = Point(3.0, 5.0, 5.0, 3.0, 1)
    p3 = Point(3.0, 6.0, 5.0, 3.0, 1)
    p4 = Point(3.0, 7.0, 5.0, 3.0, 1)
    p5 = Point(3.0, 8.0, 5.0, 3.0, 1)
    dist = path_distance([p1, p2, p3, p4, p5])
    nt.ok_(dist == 4)


def test_segment_area():
    p0 = Point(0.0, 0.0, 0.0, 3.0, 1)
    p1 = Point(2.0, 0.0, 0.0, 3.0, 1)
    p2 = Point(4.0, 0.0, 0.0, 3.0, 1)
    p3 = Point(4.0, 0.0, 0.0, 6.0, 1)
    p4 = Point(1.0, 0.0, 0.0, 3.0, 1)
    p5 = Point(4.0, 0.0, 0.0, 3.0, 1)

    a01 = segment_area((p0, p1))
    a02 = segment_area((p0, p2))
    a03 = segment_area((p0, p3))
    a04 = segment_area((p0, p4))
    a45 = segment_area((p4, p5))
    a05 = segment_area((p0, p5))

    nt.assert_almost_equal(a01, 37.6991118, places=6)
    nt.assert_almost_equal(2*a01, a02)
    nt.assert_almost_equal(a03, 141.3716694, places=6)
    nt.assert_almost_equal(a45, a05 - a04)
    nt.assert_almost_equal(segment_area((p0, p3)), segment_area((p3, p0)))


def test_segment_volume():
    p0 = Point(0.0, 0.0, 0.0, 3.0, 1)
    p1 = Point(2.0, 0.0, 0.0, 3.0, 1)
    p2 = Point(4.0, 0.0, 0.0, 3.0, 1)
    p3 = Point(4.0, 0.0, 0.0, 6.0, 1)
    p4 = Point(1.0, 0.0, 0.0, 3.0, 1)
    p5 = Point(4.0, 0.0, 0.0, 3.0, 1)

    v01 = segment_volume((p0, p1))
    v02 = segment_volume((p0, p2))
    v03 = segment_volume((p0, p3))
    v04 = segment_volume((p0, p4))
    v45 = segment_volume((p4, p5))
    v05 = segment_volume((p0, p5))

    nt.assert_almost_equal(v01, 56.5486677, places=6)
    nt.assert_almost_equal(2*v01, v02)
    nt.assert_almost_equal(v03, 263.8937829, places=6)
    nt.assert_almost_equal(v45, v05 - v04)
    nt.assert_almost_equal(segment_volume((p0, p3)), segment_volume((p3, p0)))


def test_segment_length():
    nt.ok_(segment_length(((0,0,0), (0,0,42))) == 42)
    nt.ok_(segment_length(((0,0,0), (0,42,0))) == 42)
    nt.ok_(segment_length(((0,0,0), (42,0,0))) == 42)


def test_segment_radius():
    nt.ok_(segment_radius(((0,0,0,4),(0,0,0,6))) == 5)


def test_segment_radial_dist():
    seg = ((11,11,11), (33, 33, 33))
    nt.assert_almost_equal(segment_radial_dist(seg, (0,0,0)),
                           point_dist((0,0,0), (22,22,22)))


def test_taper_rate():
    p0 = (0.0, 0.0, 0.0, 1.0)
    p1 = (1.0, 0.0, 0.0, 4.0)
    p2 = (2.0, 0.0, 0.0, 4.0)
    p3 = (3.0, 0.0, 0.0, 4.0)
    nt.assert_almost_equal(taper_rate(p0, p1), 6.0)
    nt.assert_almost_equal(taper_rate(p0, p2), 3.0)
    nt.assert_almost_equal(taper_rate(p0, p3), 2.0)


def test_segment_taper_rate():
    p0 = (0.0, 0.0, 0.0, 1.0)
    p1 = (1.0, 0.0, 0.0, 4.0)
    p2 = (2.0, 0.0, 0.0, 4.0)
    p3 = (3.0, 0.0, 0.0, 4.0)
    nt.assert_almost_equal(segment_taper_rate((p0, p1)), 6.0)
    nt.assert_almost_equal(segment_taper_rate((p0, p2)), 3.0)
    nt.assert_almost_equal(segment_taper_rate((p0, p3)), 2.0)

def test_pca():

    p = np.array([[4., 2., 0.6],
                 [4.2, 2.1, 0.59],
                 [3.9, 2.0, 0.58],
                 [4.3, 2.1, 0.62],
                 [4.1, 2.2, 0.63]])

    RES_COV = np.array([[0.025, 0.0075, 0.00175],
                        [0.0075, 0.0070, 0.00135],
                        [0.00175, 0.00135, 0.00043]])

    RES_EIGV = np.array([[ 0.93676841,  0.34958469, -0.0159843 ],
                        [ 0.34148069, -0.92313136, -0.1766902 ],
                        [ 0.0765238 , -0.16005947,  0.98413672]])

    RES_EIGS = np.array([0.0278769, 0.00439387, 0.0001592])
    eigs, eigv = pca(p)

    nt.assert_true(np.allclose(eigs, RES_EIGS))
    nt.assert_true(np.allclose(eigv[:,0], RES_EIGV[:,0]) or np.allclose(eigv[:, 0], -1. * RES_EIGV[:, 0]))
    nt.assert_true(np.allclose(eigv[:,1], RES_EIGV[:,1]) or np.allclose(eigv[:, 1], -1. * RES_EIGV[:, 1]))
    nt.assert_true(np.allclose(eigv[:,2], RES_EIGV[:,2]) or np.allclose(eigv[:, 2], -1. * RES_EIGV[:, 2]))
