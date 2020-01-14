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

import warnings
from nose import tools as nt
from neurom.core import _soma
from neurom.exceptions import SomaError
import numpy as np
import math

SOMA_SINGLE_PTS = [[11, 22, 33, 44, 1, 1, -1]]

SOMA_THREEPOINTS_PTS = np.array([
    [0, 0, 0, 44, 1, 1, -1],
    [0, -44, 0, 44, 2, 1, 1],
    [0, +44, 0, 44, 3, 1, 1],
])

SOMA_SIMPLECONTOUR_PTS_4 = [
    [1, 0, 0, 44, 1, 1, -1],
    [0, 1, 0, 44, 2, 1, 1],
    [-1, 0, 0, 44, 3, 1, 2],
    [0, -1, 0, 44, 4, 1, 3],
]

sin_pi_by_4 = math.cos(math.pi/4.)
cos_pi_by_4 = math.sin(math.pi/4.)

SOMA_SIMPLECONTOUR_PTS_6 = [
    [1, 0, 0, 44, 1, 1, -1],
    [sin_pi_by_4, cos_pi_by_4, 0, 44, 1, 1, -1],
    [0, 1, 0, 44, 2, 1, 1],
    [-1, 0, 0, 44, 3, 1, 2],
    [-sin_pi_by_4, -cos_pi_by_4, 0, 44, 1, 1, -1],
    [0, -1, 0, 44, 4, 1, 3],
]

SOMA_SIMPLECONTOUR_PTS_8 = [
    [1, 0, 0, 44, 1, 1, -1],
    [sin_pi_by_4, cos_pi_by_4, 0, 44, 1, 1, -1],
    [0, 1, 0, 44, 2, 1, 1],
    [-sin_pi_by_4, cos_pi_by_4, 0, 44, 1, 1, -1],
    [-1, 0, 0, 44, 3, 1, 2],
    [-sin_pi_by_4, -cos_pi_by_4, 0, 44, 1, 1, -1],
    [0, -1, 0, 44, 4, 1, 3],
    [sin_pi_by_4, -cos_pi_by_4, 0, 44, 1, 1, -1],
]


INVALID_PTS_0 = []

INVALID_PTS_2 = [

    [11, 22, 33, 44, 1, 1, -1],
    [11, 22, 33, 44, 2, 1, 1]
]


def test_make_Soma_SinglePoint():
    sm = _soma.make_soma(SOMA_SINGLE_PTS)
    nt.ok_('SomaSinglePoint' in str(sm))
    nt.ok_(isinstance(sm, _soma.SomaSinglePoint))
    nt.eq_(list(sm.center), [11, 22, 33])
    nt.ok_(sm.radius == 44)


def test_make_Soma_contour():
    sm = _soma.make_soma(SOMA_THREEPOINTS_PTS, soma_class=_soma.SOMA_CONTOUR)
    nt.ok_('SomaSimpleContour' in str(sm))
    nt.ok_(isinstance(sm, _soma.SomaSimpleContour))
    nt.eq_(list(sm.center), [0, 0, 0])
    nt.assert_almost_equal(sm.radius, 29.33333333, places=5)


def test_make_Soma_ThreePointCylinder():
    with warnings.catch_warnings(record=True):
        sm = _soma.make_soma(SOMA_THREEPOINTS_PTS, soma_class=_soma.SOMA_CYLINDER)
    nt.ok_('SomaNeuromorphoThreePointCylinders' in str(sm))
    nt.ok_(isinstance(sm, _soma.SomaNeuromorphoThreePointCylinders))
    nt.eq_(list(sm.center), [0, 0, 0])
    nt.eq_(sm.radius, 44)


def check_SomaC(points):
    sm = _soma.make_soma(points)
    nt.ok_('SomaSimpleContour' in str(sm))
    nt.ok_(isinstance(sm, _soma.SomaSimpleContour))
    np.testing.assert_allclose(sm.center, (0., 0., 0.), atol=1e-16)
    nt.eq_(sm.radius, 1.0)


def test_make_SomaC():
    check_SomaC(SOMA_SIMPLECONTOUR_PTS_4)
    check_SomaC(SOMA_SIMPLECONTOUR_PTS_6)
    check_SomaC(SOMA_SIMPLECONTOUR_PTS_8)


@nt.raises(SomaError)
def test_invalid_soma_points_0_raises_SomaError():
    _soma.make_soma(INVALID_PTS_0)


@nt.raises(SomaError)
def test_invalid_soma_points_2_raises_SomaError():
    _soma.make_soma(INVALID_PTS_2)


def test_make_Soma_Cylinders():
    points = [[0, 0, -10, 40],
              [0, 0,   0, 40],
              [0, 0,  10, 40],
              ]
    s = _soma.SomaCylinders(points)
    # if r = 2*h (ie: as in this case 10 - -10 == 20), then the
    # area of a cylinder (excluding end caps) is:
    # 2*pi*r*h == 4*pi*r^2 == area of a sphere of radius 20
    nt.eq_(s.radius, 20.0)
    nt.assert_almost_equal(s.area, 5026.548245743669)
    nt.eq_(s.center, [0, 0, -10])
    nt.ok_('SomaCylinders' in str(s))

    # cylinder: h = 10, r = 20
    soma_2pt_normal = np.array([
        [0.0,   0.0,  0.0, 20.0, 1, 1, -1],
        [0.0, -10.0,  0.0, 20.0, 1, 2,  1],
    ])
    s = _soma.make_soma(soma_2pt_normal, soma_class=_soma.SOMA_CYLINDER)
    nt.assert_almost_equal(s.area, 1256.6370614) # see r = 2*h above
    nt.eq_(list(s.center), [0., 0., 0.])

    #check tapering
    soma_2pt_normal = np.array([
        [0.0,   0.0,  0.0, 0.0, 1, 1, -1],
        [0.0, -10.0,  0.0, 20.0, 1, 2,  1],
    ])
    s = _soma.make_soma(soma_2pt_normal, soma_class=_soma.SOMA_CYLINDER)
    nt.assert_almost_equal(s.area, 1404.9629462081452) # cone area, not including 'bottom'

    # neuromorpho style
    soma_3pt_neuromorpho = np.array([
        [0.0,   0.0,  0.0, 10.0, 1, 1, -1],
        [0.0, -10.0,  0.0, 10.0, 1, 2,  1],
        [0.0,   10.0, 0.0, 10.0, 1, 3,  1],
    ])
    with warnings.catch_warnings(record=True):
        s = _soma.make_soma(soma_3pt_neuromorpho, soma_class=_soma.SOMA_CYLINDER)
    nt.ok_('SomaNeuromorphoThreePointCylinders' in str(s))
    nt.eq_(list(s.center), [0., 0., 0.])
    nt.assert_almost_equal(s.area, 1256.6370614)

    # some neuromorpho files don't follow the convention
    #but have (ys + rs) as point 2, and have xs different in each line
    # ex: http://neuromorpho.org/dableFiles/brumberg/CNG%20version/april11s1cell-1.CNG.swc
    soma_3pt_neuromorpho = np.array([
        [ 0.0,   0.0,  0.0, 10.0, 1, 1, -1],
        [-2.0,   6.0,  0.0, 10.0, 1, 2,  1],
        [ 2.0,  -6.0,  0.0, 10.0, 1, 3,  1],
    ])
    with warnings.catch_warnings(record=True):
        s = _soma.make_soma(soma_3pt_neuromorpho, soma_class=_soma.SOMA_CYLINDER)
    nt.ok_('SomaNeuromorphoThreePointCylinders' in str(s))
    nt.eq_(list(s.center), [0., 0., 0.])
    nt.assert_almost_equal(s.area, 794.76706126368811)

    soma_4pt_normal = np.array([
        [0.0, 0.0,  0.0, 0.0,  1, 1, -1],
        [0.0, 2.0,  0.0, 2.0,  1, 2, 1],
        [0.0, 4.0,  0.0, 4.0,  1, 3, 2],
        [0.0, 6.0,  0.0, 6.0,  1, 4, 3],
        [0.0, 8.0,  0.0, 8.0,  1, 5, 4],
        [0.0, 10.0, 0.0, 10.0, 1, 6, 5],
    ])
    s = _soma.make_soma(soma_4pt_normal, soma_class=_soma.SOMA_CYLINDER)
    nt.eq_(list(s.center), [0., 0., 0.])
    nt.assert_almost_equal(s.area, 444.288293851) # cone area, not including bottom
