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

import neurom.geom.transform as gtr
from neurom.core.dataformat import COLS
from neurom.core.tree import val_iter, ipreorder
from neurom.ezy import load_neuron
from nose import tools as nt
from itertools import izip
import numpy as np
from copy import copy


NEURON = load_neuron('test_data/valid_set/Neuron.swc')
TREE = NEURON.neurites[1]


TEST_UVEC =  np.array([ 0.01856633,  0.37132666,  0.92831665])

TEST_ANGLE = np.pi / 3.

def _Rx(angle):
    sn = np.sin(angle)
    cs = np.cos(angle)
    return np.array([[1., 0., 0.],
                     [0., cs, -sn],
                     [0., sn, cs]])


def _Ry(angle):
    sn = np.sin(angle)
    cs = np.cos(angle)
    return np.array([[cs, 0., sn],
                     [0., 1., 0.],
                     [-sn, 0., cs]])


def _Rz(angle):
    sn = np.sin(angle)
    cs = np.cos(angle)
    return np.array([[cs, -sn, 0.],
                     [sn, cs, 0.],
                     [0., 0., 1.]])


def _evaluate(tr1, tr2, comp_func):

    for v1, v2 in izip(val_iter(ipreorder(tr1)), val_iter(ipreorder(tr2))):
        #print "v1 : ", v1[:COLS.R]
        #print "v2 : ", v2[:COLS.R]
        #print "-" * 10
        nt.assert_true(comp_func(v1[:COLS.R], v2[:COLS.R]))


def test_translate_dispatch():

    nt.assert_true(isinstance(grt.translate(Neuron, np.array([1.,1.,1.])), Neuron))
    nt.assert_true(isinstance(gtr.translate(TREE, np.array([1.,1.,1.])), Tree))


def test_rotate_dispatch():

    nt.assert_true(isinstance(grt.rotate(Neuron, TEST_UVEC, np.pi) Neuron))
    nt.assert_true(isinstance(gtr.rotate(TREE, TEST_UVEC, np.pi), Tree))

def test_translate_tree():

    t = np.array([-100., 0.1, -2.4445])

    m = gtr.translate(TREE, t)

    # subtract the values node by node and assert if the changed tree values
    # minus the original result into the translation vector
    _evaluate(TREE, m, lambda x, y : np.allclose(y - x, t))


def test_rotate_tree():

    m = gtr.rotate(TREE, TEST_UVEC, TEST_ANGLE)

    R = gtr._rodriguesToRotationMatrix(TEST_UVEC, TEST_ANGLE)

    # rotation matrix inverse equals its transpose
    Rinv = R.transpose()

    # check that if the inverse rotation on the rotated result returns
    # the initial coordinates
    _evaluate(TREE, m, lambda x, y: np.allclose(np.dot(Rinv, y), x) )


    # check with origin
    new_orig = np.array([-50., 45., 30.])

    m = gtr.rotate(TREE, TEST_UVEC, TEST_ANGLE, origin=new_orig)
    m = gtr.rotate(m, TEST_UVEC, -TEST_ANGLE, origin=new_orig)

    _evaluate(TREE, m, lambda x, y: np.allclose(x, y) )


def test_rodriguesToRotationMatrix():

    RES = np.array([[0.50017235, -0.80049871, 0.33019604],
                    [0.80739289, 0.56894174, 0.15627544],
                    [-0.3129606, 0.18843328, 0.9308859]])

    R = gtr._rodriguesToRotationMatrix(TEST_UVEC, TEST_ANGLE)

    # assess rotation matrix properties:

    # detR = +=1
    nt.assert_almost_equal(np.linalg.det(R), 1.)

    # R.T = R^-1
    nt.assert_true(np.allclose(np.linalg.inv(R), R.transpose()))

    # check against calculated matrix
    nt.assert_true(np.allclose(R, RES))

    # check if opposite sign generates inverse
    Rinv = gtr._rodriguesToRotationMatrix(TEST_UVEC, -TEST_ANGLE)

    nt.assert_true(np.allclose(np.dot(Rinv, R), np.identity(3)))

    # check basic rotations with a range of angles
    for angle in np.linspace(0., 2. * np.pi, 10):

        Rx = gtr._rodriguesToRotationMatrix(np.array([1., 0., 0.]), angle)
        Ry = gtr._rodriguesToRotationMatrix(np.array([0., 1., 0.]), angle)
        Rz = gtr._rodriguesToRotationMatrix(np.array([0., 0., 1.]), angle)

        nt.assert_true(np.allclose(Rx, _Rx(angle)))
        nt.assert_true(np.allclose(Ry, _Ry(angle)))
        nt.assert_true(np.allclose(Rz, _Rz(angle)))


def test_affineTransformPoint():

    point = TREE.value[:COLS.R]
    # rotate 180 and translate, translate back and rotate 180
    # change origin as well

    new_orig = np.array([10. , 45., 50.])

    t = np.array([0.1, - 0.1, 40.3])

    R = gtr._rodriguesToRotationMatrix(TEST_UVEC, np.pi)

    m = copy(point)

    # change origin, rotate 180 and translate
    gtr._affineTransformPoint(m, R, t, origin=new_orig)

    # translate back
    gtr._affineTransformPoint(m, np.identity(3), -t, origin=np.zeros(3))

    # rotate back
    gtr._affineTransformPoint(m, R, np.zeros(3), origin=new_orig)

    nt.assert_true(np.allclose(point, m))

def test_affineTransformTree():

    # rotate 180 and translate, translate back and rotate 180
    # change origin as well

    new_orig = np.array([10. , 45., 50.])

    t = np.array([0.1, - 0.1, 40.3])

    R = gtr._rodriguesToRotationMatrix(TEST_UVEC, np.pi)

    # change origin, rotate 180 and translate
    m = gtr._affineTransform(TREE, R, t, origin=new_orig)

    # translate back
    m = gtr._affineTransform(m, np.identity(3), -t, origin=np.zeros(3))

    # rotate back
    m = gtr._affineTransform(m, R, np.zeros(3), origin=new_orig)

    _evaluate(TREE, m, lambda x, y: np.allclose(x, y))


def test_affineTransformNeuron():

    # rotate 180 and translate, translate back and rotate 180
    # change origin as well

    new_orig = np.array([10. , 45., 50.])

    t = np.array([0.1, - 0.1, 40.3])

    R = gtr._rodriguesToRotationMatrix(TEST_UVEC, np.pi)

    m = NEURON.copy()

    # change origin, rotate 180 and translate
    gtr._affineTransformNeuron(m, R, t, origin=new_orig)

    # translate back
    gtr._affineTransformNeuron(m, np.identity(3), -t, origin=np.zeros(3))

    # rotate back
    gtr._affineTransformNeuron(m, R, np.zeros(3), origin=new_orig)

    nt.asser_true(np.allclose(list(m.soma.iter()), list(NEURON.soma.iter())))

    for neu1, neu2 in izip(NEURON.neurites, m.neurites):
        _evaluate(neu1, neu2, lambda x, y: np.allclose(x, y))
