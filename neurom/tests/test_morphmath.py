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

from math import fabs, pi, sqrt

import numpy as np
from neurom import morphmath as mm
from neurom.core.point import Point
from numpy.random import uniform
from numpy.testing import assert_array_almost_equal, assert_almost_equal

np.random.seed(0)


def test_vector():
    vec1 = (12.0, 3, 6.0)
    vec2 = (14.0, 1.0, 6.0)
    vec = mm.vector(vec1, vec2)
    assert np.all(vec == (-2.0, 2.0, 0.0))


def test_linear_interpolate():
    p0 = np.array([-1., -1., -1.])
    p1 = np.array([1., 1., 1.])

    res = mm.linear_interpolate(p0, p1, 0.0)
    assert np.allclose(res, (-1., -1., -1.))
    res = mm.linear_interpolate(p0, p1, 0.25)
    assert np.allclose(res, (-0.5, -0.5, -0.5))
    res = mm.linear_interpolate(p0, p1, 0.5)
    assert np.allclose(res, (0., 0., 0.))
    res = mm.linear_interpolate(p0, p1, 0.75)
    assert np.allclose(res, (0.5, 0.5, 0.5))
    res = mm.linear_interpolate(p0, p1, 1.0)
    assert np.allclose(res, (1., 1., 1.))


def test_interpolate_radius_r1_g_r2():
    res = mm.interpolate_radius(2., 1., 0.1)
    assert res == 1.9


def test_interpolate_radius_r2_g_r1():
    res = mm.interpolate_radius(1., 2., 0.2)
    assert res == 1.2


def test_interpolate_radius_extreme_cases():
    res = mm.interpolate_radius(1., 1., 0.2)
    assert res == 1.
    res = mm.interpolate_radius(0., 2., 0.3)
    assert res == 2. * 0.3
    res = mm.interpolate_radius(3., 0., 0.15)
    assert res == 3. * (1. - 0.15)


def test_path_fraction_point_two_points():

    points = [np.array([-1., -1., -1.]), np.array([1., 1., 1.])]

    res = mm.path_fraction_point(points, 0.0)
    assert np.allclose(res, (-1., -1., -1.))
    res = mm.path_fraction_point(points, 0.25)
    assert np.allclose(res, (-0.5, -0.5, -0.5))
    res = mm.path_fraction_point(points, 1.0)
    assert np.allclose(res, (1., 1., 1.))


def test_path_fraction_three_symmetric_points():

    points = [np.array((1., 0., 0.)),
              np.array((0., 0., 0.)),
              np.array((0., 0., 1.))]

    res = mm.path_fraction_point(points, 0.0)
    assert np.allclose(res, (1., 0., 0.))

    res = mm.path_fraction_point(points, 0.25)
    assert np.allclose(res, (0.5, 0., 0.))

    res = mm.path_fraction_point(points, 0.5)
    assert np.allclose(res, (0., 0., 0.))

    res = mm.path_fraction_point(points, 0.75)
    assert np.allclose(res, (0., 0., 0.5))

    res = mm.path_fraction_point(points, 1.0)
    assert np.allclose(res, (0., 0., 1.))


def test_path_fraction_many_points():
    def x(theta): return np.cos(theta)
    def y(theta): return np.sin(theta)
    points = [np.array((x(theta), y(theta), 2.))
              for theta in (0., np.pi/4., np.pi/2., 3.*np.pi/4., np.pi)]

    res = mm.path_fraction_point(points, 0.)
    assert np.allclose(res, (x(0.), y(0.), 2.))

    res = mm.path_fraction_point(points, 0.25)
    assert np.allclose(res, (x(np.pi / 4.), y(np.pi / 4.), 2.))

    res = mm.path_fraction_point(points, 0.5)
    assert np.allclose(res, (x(np.pi / 2.), y(np.pi / 2.), 2.))

    res = mm.path_fraction_point(points, 0.75)
    assert np.allclose(res, (x(3. * np.pi / 4.), y(3. * np.pi / 4.), 2.))

    res = mm.path_fraction_point(points, 1.)
    assert np.allclose(res, (x(np.pi), y(np.pi), 2.))


def test_scalar_projection():
    v1 = np.array([4., 1., 0.])
    v2 = np.array([2., 3., 0.])

    res = mm.scalar_projection(v1, v2)
    assert np.isclose(res, 3.0508510792387602)


def test_scalar_projection_collinear():
    v1 = np.array([1., 2., 0.])
    v2 = np.array([4., 8., 0.])

    res = mm.scalar_projection(v1, v2)

    assert np.allclose(res, 20. / np.linalg.norm(v2))


def test_scalar_projection_perpendicular():
    v1 = np.array([3., 0., 0.])
    v2 = np.array([0., 1.5, 0.])

    res = mm.scalar_projection(v1, v2)
    assert np.allclose(res, 0.)


def test_vector_projection():
    v1 = np.array([4., 1., 0.])
    v2 = np.array([2., 3., 0.])

    res = mm.vector_projection(v1, v2)
    assert np.allclose(res, (1.6923076923076923, 2.5384615384615383, 0.))


def test_vector_projection_collinear():

    v1 = np.array([1., 2., 3.])
    v2 = np.array([4., 8., 12.])

    res = mm.vector_projection(v1, v2)
    assert np.allclose(res, v1)


def test_vector_projection_perpendicular():

    v1 = np.array([2., 0., 0.])
    v2 = np.array([0., 3., 0.])

    res = mm.vector_projection(v1, v2)
    assert np.allclose(res, (0., 0., 0.))


def test_dist_point_line():

    # an easy one:
    res = mm.dist_point_line(np.array([0., 0., 0.]), np.array([0., 1., 0.]), np.array([1., 0., 0.]))
    assert np.isclose(res, np.sqrt(2) / 2.)

    # check the distance of the line 3x - 4y + 1 = 0
    # with parametric form of (t, (4t - 1)/3)
    # two points that satisfy this equation:
    l1 = np.array([0., 1. / 4., 0.])
    l2 = np.array([1., 1., 0.])

    p = np.array([2., 3., 0.])

    res = mm.dist_point_line(p, l1, l2)
    assert res == 1.


def test_point_dist2():
    p1 = Point(3.0, 4.0, 5.0, 3.0)
    p2 = Point(4.0, 5.0, 6.0, 3.0)
    dist = mm.point_dist2(p1, p2)
    assert dist == 3


def test_segment_length2():
    p1 = Point(3.0, 4.0, 5.0, 3.0)
    p2 = Point(4.0, 5.0, 6.0, 3.0)
    dist = mm.segment_length2((p1, p2))
    assert dist == 3


def test_point_dist():
    p1 = Point(3.0, 4.0, 5.0, 3.0)
    p2 = Point(4.0, 5.0, 6.0, 3.0)
    dist = mm.point_dist(p1, p2)
    assert dist == sqrt(3)


def test_angle_3points_half_pi():
    orig = (0.0, 0.0, 0.0)
    vec1 = (1.0, 0.0, 0.0)
    vec2 = (0.0, 2.0, 0.0)
    assert mm.angle_3points(orig, vec1, vec2) == pi / 2.0

    vec2 = (0.0, 0.0, 3.0)
    assert mm.angle_3points(orig, vec1, vec2) == pi / 2.0

    vec2 = (0.0, 0.0, -3.0)
    assert mm.angle_3points(orig, vec1, vec2) == pi / 2.0

    vec1 = (0.0, 4.0, 0.0)
    assert mm.angle_3points(orig, vec1, vec2) == pi / 2.0


def test_angle_3points_quarter_pi():
    orig = (0.0, 0.0, 0.0)
    vec1 = (1.0, 0.0, 0.0)
    vec2 = (2.0, 2.0, 0.0)
    assert mm.angle_3points(orig, vec1, vec2) == pi / 4.0

    vec2 = (3.0, 3.0, 0.0)
    assert mm.angle_3points(orig, vec1, vec2) == pi / 4.0

    vec2 = (3.0, -3.0, 0.0)
    assert mm.angle_3points(orig, vec1, vec2) == pi / 4.0

    vec2 = (3.0, 0.0, 3.0)
    assert mm.angle_3points(orig, vec1, vec2) == pi / 4.0

    vec2 = (3.0, 0.0, -3.0)
    assert mm.angle_3points(orig, vec1, vec2) == pi / 4.0


def test_angle_3points_three_quarter_pi():
    orig = (0.0, 0.0, 0.0)
    vec1 = (1.0, 0.0, 0.0)
    vec2 = (-2.0, 2.0, 0.0)
    assert mm.angle_3points(orig, vec1, vec2) == 3 * pi / 4.0

    vec2 = (-3.0, 3.0, 0.0)
    assert mm.angle_3points(orig, vec1, vec2) == 3 * pi / 4.0

    vec2 = (-3.0, -3.0, 0.0)
    assert mm.angle_3points(orig, vec1, vec2) == 3 * pi / 4.0

    vec2 = (-3.0, 0.0, 3.0)
    assert mm.angle_3points(orig, vec1, vec2) == 3 * pi / 4.0

    vec2 = (-3.0, 0.0, -3.0)
    assert mm.angle_3points(orig, vec1, vec2) == 3 * pi / 4.0


def test_angle_3points_equal_points_returns_zero():
    orig = (0.0, 1.0, 0.0)
    vec1 = (1.0, 0.0, 0.0)
    vec2 = (0.0, 1.0, 0.0)
    a = mm.angle_3points(orig, vec1, vec2)
    assert a == 0.0


def test_angle_3points_opposing_returns_pi():
    orig = (0.0, 0.0, 0.0)
    vec1 = (1.0, 1.0, 1.0)
    vec2 = (-2.0, -2.0, -2.0)
    angle = mm.angle_3points(orig, vec1, vec2)
    assert angle == pi


def test_angle_3points_collinear_returns_zero():
    orig = (0.0, 0.0, 0.0)
    vec1 = (1.0, 1.0, 1.0)
    vec2 = (2.0, 2.0, 2.0)
    angle = mm.angle_3points(orig, vec1, vec2)
    assert angle == 0.0


def test_angle_between_vectors():
    angle1 = mm.angle_between_vectors((1, 0), (0, 1))
    assert angle1 == np.pi/2
    angle1 = mm.angle_between_vectors((1, 0), (1, 0))
    assert angle1 == 0.0
    angle1 = mm.angle_between_vectors((1, 0), (-1, 0))
    assert angle1 == np.pi


def soma_points(radius=5, number_points=20):
    phi = uniform(0, 2*pi, number_points)
    costheta = uniform(-1, 1, number_points)
    theta = np.arccos(costheta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return [Point(i, j, k, 0.0) for (i, j, k) in zip(x, y, z)]


def test_polygon_diameter():
    p1 = Point(3.0, 4.0, 5.0, 3.0)
    p2 = Point(3.0, 5.0, 5.0, 3.0)
    p3 = Point(3.0, 6.0, 5.0, 3.0)
    dia = mm.polygon_diameter([p1, p2, p3])
    assert dia == 2.0
    surfpoint = soma_points()
    dia1 = mm.polygon_diameter(surfpoint)
    assert fabs(dia1-10.0) < 0.1


def test_average_points_dist():
    p0 = Point(0.0, 0.0, 0.0, 3.0)
    p1 = Point(0.0, 0.0, 1.0, 3.0)
    p2 = Point(0.0, 1.0, 0.0, 3.0)
    p3 = Point(1.0, 0.0, 0.0, 3.0)
    av_dist = mm.average_points_dist(p0, [p1, p2, p3])
    assert av_dist == 1.0


def test_path_distance():
    p1 = Point(3.0, 4.0, 5.0, 3.0)
    p2 = Point(3.0, 5.0, 5.0, 3.0)
    p3 = Point(3.0, 6.0, 5.0, 3.0)
    p4 = Point(3.0, 7.0, 5.0, 3.0)
    p5 = Point(3.0, 8.0, 5.0, 3.0)
    dist = mm.path_distance([p1, p2, p3, p4, p5])
    assert dist == 4


def test_segment_area():
    p0 = Point(0.0, 0.0, 0.0, 3.0)
    p1 = Point(2.0, 0.0, 0.0, 3.0)
    p2 = Point(4.0, 0.0, 0.0, 3.0)
    p3 = Point(4.0, 0.0, 0.0, 6.0)
    p4 = Point(1.0, 0.0, 0.0, 3.0)
    p5 = Point(4.0, 0.0, 0.0, 3.0)

    a01 = mm.segment_area((p0, p1))
    a02 = mm.segment_area((p0, p2))
    a03 = mm.segment_area((p0, p3))
    a04 = mm.segment_area((p0, p4))
    a45 = mm.segment_area((p4, p5))
    a05 = mm.segment_area((p0, p5))

    assert_almost_equal(a01, 37.6991118, decimal=6)
    assert_almost_equal(2 * a01, a02)
    assert_almost_equal(a03, 141.3716694, decimal=6)
    assert_almost_equal(a45, a05 - a04)
    assert_almost_equal(mm.segment_area((p0, p3)), mm.segment_area((p3, p0)))


def test_segment_volume():
    p0 = Point(0.0, 0.0, 0.0, 3.0)
    p1 = Point(2.0, 0.0, 0.0, 3.0)
    p2 = Point(4.0, 0.0, 0.0, 3.0)
    p3 = Point(4.0, 0.0, 0.0, 6.0)
    p4 = Point(1.0, 0.0, 0.0, 3.0)
    p5 = Point(4.0, 0.0, 0.0, 3.0)

    v01 = mm.segment_volume((p0, p1))
    v02 = mm.segment_volume((p0, p2))
    v03 = mm.segment_volume((p0, p3))
    v04 = mm.segment_volume((p0, p4))
    v45 = mm.segment_volume((p4, p5))
    v05 = mm.segment_volume((p0, p5))

    assert_almost_equal(v01, 56.5486677, decimal=6)
    assert_almost_equal(2 * v01, v02)
    assert_almost_equal(v03, 263.8937829, decimal=6)
    assert_almost_equal(v45, v05 - v04)
    assert_almost_equal(mm.segment_volume((p0, p3)), mm.segment_volume((p3, p0)))


def test_segment_length():
    assert mm.segment_length(((0, 0, 0), (0, 0, 42))) == 42
    assert mm.segment_length(((0, 0, 0), (0, 42, 0))) == 42
    assert mm.segment_length(((0, 0, 0), (42, 0, 0))) == 42


def test_segment_radius():
    assert mm.segment_radius(((0, 0, 0, 4), (0, 0, 0, 6))) == 5


def test_segment_x_coordinate():
    assert mm.segment_x_coordinate(((0, 0, 0, 4), (0, 0, 0, 6))) == 0
    assert mm.segment_x_coordinate(((0, 0, 0, 4), (1, 0, 0, 6))) == 0.5


def test_segment_y_coordinate():
    assert mm.segment_y_coordinate(((0, 0, 0, 4), (0, 0, 0, 6))) == 0
    assert mm.segment_y_coordinate(((0, 0, 0, 4), (0, 1, 0, 6))) == 0.5


def test_segment_z_coordinate():
    assert mm.segment_z_coordinate(((0, 0, 0, 4), (0, 0, 0, 6))) == 0
    assert mm.segment_z_coordinate(((0, 0, 0, 4), (0, 0, 1, 6))) == 0.5


def test_segment_radial_dist():
    seg = ((11, 11, 11), (33, 33, 33))
    assert_almost_equal(mm.segment_radial_dist(seg, (0, 0, 0)),
                           mm.point_dist((0, 0, 0), (22, 22, 22)))


def test_taper_rate():
    p0 = (0.0, 0.0, 0.0, 1.0)
    p1 = (1.0, 0.0, 0.0, 4.0)
    p2 = (2.0, 0.0, 0.0, 4.0)
    p3 = (3.0, 0.0, 0.0, 4.0)
    assert_almost_equal(mm.taper_rate(p0, p1), 6.0)
    assert_almost_equal(mm.taper_rate(p0, p2), 3.0)
    assert_almost_equal(mm.taper_rate(p0, p3), 2.0)


def test_segment_taper_rate():
    p0 = (0.0, 0.0, 0.0, 1.0)
    p1 = (1.0, 0.0, 0.0, 4.0)
    p2 = (2.0, 0.0, 0.0, 4.0)
    p3 = (3.0, 0.0, 0.0, 4.0)
    assert_almost_equal(mm.segment_taper_rate((p0, p1)), 6.0)
    assert_almost_equal(mm.segment_taper_rate((p0, p2)), 3.0)
    assert_almost_equal(mm.segment_taper_rate((p0, p3)), 2.0)


def test_pca():

    p = np.array([[4., 2., 0.6],
                  [4.2, 2.1, 0.59],
                  [3.9, 2.0, 0.58],
                  [4.3, 2.1, 0.62],
                  [4.1, 2.2, 0.63]])

    RES_COV = np.array([[0.025, 0.0075, 0.00175],
                        [0.0075, 0.0070, 0.00135],
                        [0.00175, 0.00135, 0.00043]])

    RES_EIGV = np.array([[0.93676841,  0.34958469, -0.0159843],
                         [0.34148069, -0.92313136, -0.1766902],
                         [0.0765238, -0.16005947,  0.98413672]])

    RES_EIGS = np.array([0.0278769, 0.00439387, 0.0001592])
    eigs, eigv = mm.pca(p)

    assert np.allclose(eigs, RES_EIGS)
    assert (np.allclose(eigv[:, 0], RES_EIGV[:, 0])
                   or np.allclose(eigv[:, 0], -1. * RES_EIGV[:, 0]))
    assert (np.allclose(eigv[:, 1], RES_EIGV[:, 1])
                   or np.allclose(eigv[:, 1], -1. * RES_EIGV[:, 1]))
    assert (np.allclose(eigv[:, 2], RES_EIGV[:, 2])
                   or np.allclose(eigv[:, 2], -1. * RES_EIGV[:, 2]))


def test_sphere_area():
    area = mm.sphere_area(0.5)
    assert_almost_equal(area, pi)


def test_interval_lengths():
    assert_array_almost_equal(mm.interval_lengths([[0, 0, 0], [1, 1, 0], [2, 11, 0]]),
                              [1.414214, 10.049876])

    assert_array_almost_equal(mm.interval_lengths([[0, 0, 0], [1, 1, 0], [2, 11, 0]],
                                                  prepend_zero=True),
                              [0, 1.414214, 10.049876])
