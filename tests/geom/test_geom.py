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

from pathlib import Path

import neurom as nm
import numpy as np
from neurom import geom
from numpy.testing import assert_almost_equal

SWC_DATA_PATH = Path(__file__).parent.parent / 'data/swc'
NRN = nm.load_neuron(SWC_DATA_PATH / 'Neuron.swc')
SIMPLE = nm.load_neuron(SWC_DATA_PATH / 'simple.swc')

class PointObj:
    pass


def test_bounding_box():

    pts = np.array([[-1, -2, -3, -999],
                    [1, 2, 3, 1000],
                    [-100, 5, 33, 42],
                    [42, 55, 12, -3]])

    obj = PointObj()
    obj.points = pts

    assert np.alltrue(geom.bounding_box(obj) == [[-100, -2, -3], [42, 55, 33]])


def test_bounding_box_neuron():

    ref = np.array([[-40.32853516, -57.600172, 0.],
                    [64.74726272, 48.51626225, 54.20408797]])

    assert np.allclose(geom.bounding_box(NRN), ref)


def test_bounding_box_soma():
    ref = np.array([[0., 0., 0.], [0.1, 0.2, 0.]])
    assert np.allclose(geom.bounding_box(NRN.soma), ref)


def test_bounding_box_neurite():
    nrt = SIMPLE.neurites[0]
    ref = np.array([[-5.,  0.,  0.], [ 6.,  5.,  0.]])
    np.testing.assert_allclose(geom.bounding_box(nrt), ref)

def test_convex_hull_points():

    # This leverages scipy ConvexHull and we don't want
    # to re-test scipy, so simply check that the points are the same.
    hull = geom.convex_hull(NRN)
    assert np.alltrue(hull.points == NRN.points[:, :3])


def test_convex_hull_volume():

    # This leverages scipy ConvexHull and we don't want
    # to re-test scipy, so simply regression test the volume
    hull = geom.convex_hull(NRN)
    assert_almost_equal(hull.volume, 208641, decimal=0)
