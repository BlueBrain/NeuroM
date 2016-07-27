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
from neurom.core import soma
from neurom.exceptions import SomaError
import numpy as np
import math

SOMA_SINGLE_PTS = [[11, 22, 33, 44, 1, 1, -1]]

SOMA_THREEPOINTS_PTS = [
    [11, 22, 33, 44, 1, 1, -1],
    [11, 22, 33, 44, 2, 1, 1],
    [11, 22, 33, 44, 3, 1, 2],
]

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
    sm = soma.make_soma(SOMA_SINGLE_PTS)
    nt.ok_('SomaSinglePoint' in str(sm))
    nt.ok_(isinstance(sm, soma.SomaSinglePoint))
    nt.assert_items_equal(sm.center, (11, 22, 33))
    nt.ok_(sm.radius == 44)


def test_make_Soma_ThreePoint():
    sm = soma.make_soma(SOMA_THREEPOINTS_PTS)
    nt.ok_('SomaThreePoint' in str(sm))
    nt.ok_(isinstance(sm, soma.SomaThreePoint))
    nt.assert_items_equal(sm.center, (11, 22, 33))
    nt.eq_(sm.radius, 0.0)


def check_SomaC(points):
    sm = soma.make_soma(points)
    nt.ok_('SomaSimpleContour' in str(sm))
    nt.ok_(isinstance(sm, soma.SomaSimpleContour))
    np.testing.assert_allclose(sm.center, (0., 0., 0.), atol=1e-16)
    nt.eq_(sm.radius, 1.0)


def test_make_SomaC():
    check_SomaC(SOMA_SIMPLECONTOUR_PTS_4)
    check_SomaC(SOMA_SIMPLECONTOUR_PTS_6)
    check_SomaC(SOMA_SIMPLECONTOUR_PTS_8)


@nt.raises(SomaError)
def test_invalid_soma_points_0_raises_SomaError():
    soma.make_soma(INVALID_PTS_0)


@nt.raises(SomaError)
def test_invalid_soma_points_2_raises_SomaError():
    soma.make_soma(INVALID_PTS_2)
