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

import math
from pathlib import Path

import neurom.geom.transform as gtr
import numpy as np
from neurom import load_neuron
from neurom.fst import _neuritefunc as _nf
from nose import tools as nt

TEST_UVEC = np.array([0.01856633,  0.37132666,  0.92831665])

TEST_ANGLE = np.pi / 3.

DATA_PATH = Path(__file__).parent.parent.parent.parent / 'test_data'
H5_NRN_PATH = DATA_PATH / 'h5/v1/Neuron.h5'
SWC_NRN_PATH = DATA_PATH / 'swc/Neuron.swc'


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


@nt.raises(NotImplementedError)
def test_not_implemented_transform_call_raises():
    class Dummy(gtr.Transform3D):
        pass

    d = Dummy()
    d([1, 2, 3])


@nt.raises(NotImplementedError)
def test_translate_bad_type_raises():
    gtr.translate("hello", [1, 2, 3])


@nt.raises(NotImplementedError)
def test_rotate_bad_type_raises():
    gtr.rotate("hello", [1, 0, 0], math.pi)


def test_translate_point():

    t = gtr.Translation([100, -100, 100])
    point = [1, 2, 3]
    nt.assert_equal(t(point).tolist(), [101, -98, 103])


def test_translate_points():

    t = gtr.Translation([100, -100, 100])
    points = np.array([[1, 2, 3], [11, 22, 33], [111, 222, 333]])
    nt.assert_true(np.all(t(points) == np.array([[101, -98, 103],
                                                 [111, -78, 133],
                                                 [211, 122, 433]])))


ROT_90 = np.array([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])

ROT_180 = np.array([[-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]])

ROT_270 = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]])


def test_rotate_point():

    rot = gtr.Rotation(ROT_90)
    nt.assert_equal(rot([2, 0, 0]).tolist(), [0, 2, 0])
    nt.assert_equal(rot([0, 2, 0]).tolist(), [-2, 0, 0])
    nt.assert_equal(rot([0, 0, 2]).tolist(), [0, 0, 2])

    rot = gtr.Rotation(ROT_180)
    nt.assert_equal(rot([2, 0, 0]).tolist(), [-2, 0, 0])
    nt.assert_equal(rot([0, 2, 0]).tolist(), [0, -2, 0])
    nt.assert_equal(rot([0, 0, 2]).tolist(), [0, 0, 2])

    rot = gtr.Rotation(ROT_270)
    nt.assert_equal(rot([2, 0, 0]).tolist(), [0, -2, 0])
    nt.assert_equal(rot([0, 2, 0]).tolist(), [2, 0, 0])
    nt.assert_equal(rot([0, 0, 2]).tolist(), [0, 0, 2])


def test_rotate_points():

    rot = gtr.Rotation(ROT_90)

    points = np.array([[2, 0, 0],
                       [0, 2, 0],
                       [0, 0, 2],
                       [3, 0, 3]])

    nt.assert_true(np.all(rot(points) == np.array([[0, 2, 0],
                                                   [-2, 0, 0],
                                                   [0, 0, 2],
                                                   [0, 3, 3]])))

    rot = gtr.Rotation(ROT_180)
    nt.assert_true(np.all(rot(points) == np.array([[-2, 0, 0],
                                                   [0, -2, 0],
                                                   [0, 0, 2],
                                                   [-3, 0, 3]])))

    rot = gtr.Rotation(ROT_270)
    nt.assert_true(np.all(rot(points) == np.array([[0, -2, 0],
                                                   [2, 0, 0],
                                                   [0, 0, 2],
                                                   [0, -3, 3]])))


def test_pivot_rotate_point():

    point = [1, 2, 3]

    new_orig = np.array([10., 45., 50.])

    t = gtr.Translation(new_orig)
    t_inv = gtr.Translation(new_orig * -1)

    R = gtr._rodrigues_to_dcm(TEST_UVEC, np.pi)

    # change origin, rotate 180
    p1 = gtr.PivotRotation(R, new_orig)(point)

    # do the steps manually
    p2 = t_inv(point)
    p2 = gtr.Rotation(R)(p2)
    p2 = t(p2)

    nt.assert_equal(p1.tolist(), p2.tolist())


def test_pivot_rotate_points():

    points = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12]])

    new_orig = np.array([10., 45., 50.])

    t = gtr.Translation(new_orig)
    t_inv = gtr.Translation(new_orig * -1)

    R = gtr._rodrigues_to_dcm(TEST_UVEC, np.pi)

    # change origin, rotate 180
    p1 = gtr.PivotRotation(R, new_orig)(points)

    # do the steps manually
    p2 = t_inv(points)
    p2 = gtr.Rotation(R)(p2)
    p2 = t(p2)

    nt.assert_true(np.all(p1 == p2))


def _check_fst_nrn_translate(nrn_a, nrn_b, t):

    # soma points
    nt.assert_true(np.allclose((nrn_b.soma.points[:, 0:3] - nrn_a.soma.points[:, 0:3]), t))

    _check_fst_neurite_translate(nrn_a.neurites, nrn_b.neurites, t)


def _check_fst_neurite_translate(nrts_a, nrts_b, t):
    # neurite sections
    for sa, sb in zip(_nf.iter_sections(nrts_a),
                      _nf.iter_sections(nrts_b)):
        nt.assert_true(np.allclose((sb.points[:, 0:3] - sa.points[:, 0:3]), t))


def test_translate_fst_neuron_swc():

    t = np.array([100., 100., 100.])
    nrn = load_neuron(SWC_NRN_PATH)
    tnrn = gtr.translate(nrn, t)
    _check_fst_nrn_translate(nrn, tnrn, t)


def test_translate_fst_neurite_swc():

    t = np.array([100., 100., 100.])
    nrn = load_neuron(SWC_NRN_PATH)
    nrt_a = nrn.neurites[0]
    nrt_b = gtr.translate(nrt_a, t)
    _check_fst_neurite_translate(nrt_a, nrt_b, t)


def test_transform_translate_neuron_swc():
    t = np.array([100., 100., 100.])
    nrn = load_neuron(SWC_NRN_PATH)
    tnrn = nrn.transform(gtr.Translation(t))
    _check_fst_nrn_translate(nrn, tnrn, t)


def test_translate_fst_neuron_h5():

    t = np.array([100., 100., 100.])
    nrn = load_neuron(H5_NRN_PATH)
    tnrn = gtr.translate(nrn, t)

    _check_fst_nrn_translate(nrn, tnrn, t)


def test_translate_fst_neurite_h5():

    t = np.array([100., 100., 100.])
    nrn = load_neuron(H5_NRN_PATH)
    nrt_a = nrn.neurites[0]
    nrt_b = gtr.translate(nrt_a, t)
    _check_fst_neurite_translate(nrt_a, nrt_b, t)


def test_transform_translate_neuron_h5():
    t = np.array([100., 100., 100.])
    nrn = load_neuron(H5_NRN_PATH)
    tnrn = nrn.transform(gtr.Translation(t))
    _check_fst_nrn_translate(nrn, tnrn, t)


def _apply_rot(points, rot_mat):
    return np.dot(rot_mat, np.array(points).T).T


def _check_fst_nrn_rotate(nrn_a, nrn_b, rot_mat):

    # soma points
    nt.assert_true(np.allclose(_apply_rot(nrn_a.soma.points[:, 0:3], rot_mat),
                               nrn_b.soma.points[:, 0:3]))

    # neurite sections
    _check_fst_neurite_rotate(nrn_a.neurites, nrn_b.neurites, rot_mat)


def _check_fst_neurite_rotate(nrt_a, nrt_b, rot_mat):
    for sa, sb in zip(_nf.iter_sections(nrt_a),
                      _nf.iter_sections(nrt_b)):
        nt.assert_true(np.allclose(sb.points[:, 0:3],
                                   _apply_rot(sa.points[:, 0:3], rot_mat)))


def test_rotate_neuron_swc():
    nrn_a = load_neuron(SWC_NRN_PATH)
    nrn_b = gtr.rotate(nrn_a, [0, 0, 1], math.pi/2.0)
    rot = gtr._rodrigues_to_dcm([0, 0, 1], math.pi/2.0)
    _check_fst_nrn_rotate(nrn_a, nrn_b, rot)


def test_rotate_neurite_swc():
    nrn_a = load_neuron(SWC_NRN_PATH)
    nrt_a = nrn_a.neurites[0]
    nrt_b = gtr.rotate(nrt_a, [0, 0, 1], math.pi/2.0)
    rot = gtr._rodrigues_to_dcm([0, 0, 1], math.pi/2.0)
    _check_fst_neurite_rotate(nrt_a, nrt_b, rot)


def test_transform_rotate_neuron_swc():
    rot = gtr.Rotation(ROT_90)
    nrn_a = load_neuron(SWC_NRN_PATH)
    nrn_b = nrn_a.transform(rot)
    _check_fst_nrn_rotate(nrn_a, nrn_b, ROT_90)


def test_rotate_neuron_h5():
    nrn_a = load_neuron(H5_NRN_PATH)
    nrn_b = gtr.rotate(nrn_a, [0, 0, 1], math.pi/2.0)
    rot = gtr._rodrigues_to_dcm([0, 0, 1], math.pi/2.0)
    _check_fst_nrn_rotate(nrn_a, nrn_b, rot)


def test_rotate_neurite_h5():
    nrn_a = load_neuron(H5_NRN_PATH)
    nrt_a = nrn_a.neurites[0]
    nrt_b = gtr.rotate(nrt_a, [0, 0, 1], math.pi/2.0)
    rot = gtr._rodrigues_to_dcm([0, 0, 1], math.pi/2.0)
    _check_fst_neurite_rotate(nrt_a, nrt_b, rot)


def test_transform_rotate_neuron_h5():
    rot = gtr.Rotation(ROT_90)
    nrn_a = load_neuron(H5_NRN_PATH)
    nrn_b = nrn_a.transform(rot)
    _check_fst_nrn_rotate(nrn_a, nrn_b, ROT_90)


def test_rodrigues_to_dcm():

    RES = np.array([[0.50017235, -0.80049871, 0.33019604],
                    [0.80739289, 0.56894174, 0.15627544],
                    [-0.3129606, 0.18843328, 0.9308859]])

    R = gtr._rodrigues_to_dcm(TEST_UVEC, TEST_ANGLE)

    # assess rotation matrix properties:

    # detR = +=1
    nt.assert_almost_equal(np.linalg.det(R), 1.)

    # R.T = R^-1
    nt.assert_true(np.allclose(np.linalg.inv(R), R.transpose()))

    # check against calculated matrix
    nt.assert_true(np.allclose(R, RES))

    # check if opposite sign generates inverse
    Rinv = gtr._rodrigues_to_dcm(TEST_UVEC, -TEST_ANGLE)

    nt.assert_true(np.allclose(np.dot(Rinv, R), np.identity(3)))

    # check basic rotations with a range of angles
    for angle in np.linspace(0., 2. * np.pi, 10):

        Rx = gtr._rodrigues_to_dcm(np.array([1., 0., 0.]), angle)
        Ry = gtr._rodrigues_to_dcm(np.array([0., 1., 0.]), angle)
        Rz = gtr._rodrigues_to_dcm(np.array([0., 0., 1.]), angle)

        nt.assert_true(np.allclose(Rx, _Rx(angle)))
        nt.assert_true(np.allclose(Ry, _Ry(angle)))
        nt.assert_true(np.allclose(Rz, _Rz(angle)))
