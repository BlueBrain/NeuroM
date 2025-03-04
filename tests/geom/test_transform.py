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
import morphio
from pathlib import Path

import neurom.geom.transform as gtr
import numpy as np
from neurom import COLS, load_morphology, iter_sections

import pytest
from numpy.testing import assert_almost_equal

TEST_UVEC = np.array([0.01856633, 0.37132666, 0.92831665])
TEST_ANGLE = np.pi / 3.0
DATA_PATH = Path(__file__).parent.parent / 'data'
H5_NRN_PATH = DATA_PATH / 'h5/v1/Neuron.h5'
SWC_NRN_PATH = DATA_PATH / 'swc/Neuron.swc'


def _Rx(angle):
    sn = np.sin(angle)
    cs = np.cos(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cs, -sn], [0.0, sn, cs]])


def _Ry(angle):
    sn = np.sin(angle)
    cs = np.cos(angle)
    return np.array([[cs, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, cs]])


def _Rz(angle):
    sn = np.sin(angle)
    cs = np.cos(angle)
    return np.array([[cs, -sn, 0.0], [sn, cs, 0.0], [0.0, 0.0, 1.0]])


def test_not_implemented_transform_call_raises():
    with pytest.raises(NotImplementedError):

        class Dummy(gtr.Transform3D):
            pass

        d = Dummy()
        d([1, 2, 3])


def test_translate_bad_type_raises():
    with pytest.raises(NotImplementedError):
        gtr.translate("hello", [1, 2, 3])


def test_rotate_bad_type_raises():
    with pytest.raises(NotImplementedError):
        gtr.rotate("hello", [1, 0, 0], math.pi)


def test_translate_point():
    t = gtr.Translation([100, -100, 100])
    point = [1, 2, 3]
    assert t(point).tolist() == [101, -98, 103]


def test_translate_points():
    t = gtr.Translation([100, -100, 100])
    points = np.array([[1, 2, 3], [11, 22, 33], [111, 222, 333]])
    assert np.all(t(points) == np.array([[101, -98, 103], [111, -78, 133], [211, 122, 433]]))


ROT_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

ROT_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

ROT_270 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])


def test_rotate_point():
    rot = gtr.Rotation(ROT_90)
    assert rot([2, 0, 0]).tolist() == [0, 2, 0]
    assert rot([0, 2, 0]).tolist() == [-2, 0, 0]
    assert rot([0, 0, 2]).tolist() == [0, 0, 2]

    rot = gtr.Rotation(ROT_180)
    assert rot([2, 0, 0]).tolist() == [-2, 0, 0]
    assert rot([0, 2, 0]).tolist() == [0, -2, 0]
    assert rot([0, 0, 2]).tolist() == [0, 0, 2]

    rot = gtr.Rotation(ROT_270)
    assert rot([2, 0, 0]).tolist() == [0, -2, 0]
    assert rot([0, 2, 0]).tolist() == [2, 0, 0]
    assert rot([0, 0, 2]).tolist() == [0, 0, 2]


def test_rotate_points():
    rot = gtr.Rotation(ROT_90)

    points = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2], [3, 0, 3]])

    assert np.all(rot(points) == np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 2], [0, 3, 3]]))

    rot = gtr.Rotation(ROT_180)
    assert np.all(rot(points) == np.array([[-2, 0, 0], [0, -2, 0], [0, 0, 2], [-3, 0, 3]]))

    rot = gtr.Rotation(ROT_270)
    assert np.all(rot(points) == np.array([[0, -2, 0], [2, 0, 0], [0, 0, 2], [0, -3, 3]]))


def test_pivot_rotate_point():
    point = [1, 2, 3]

    new_orig = np.array([10.0, 45.0, 50.0])

    t = gtr.Translation(new_orig)
    t_inv = gtr.Translation(new_orig * -1)

    R = gtr._rodrigues_to_dcm(TEST_UVEC, np.pi)

    # change origin, rotate 180
    p1 = gtr.PivotRotation(R, new_orig)(point)

    # do the steps manually
    p2 = t_inv(point)
    p2 = gtr.Rotation(R)(p2)
    p2 = t(p2)

    assert p1.tolist() == p2.tolist()


def test_pivot_rotate_points():
    points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    new_orig = np.array([10.0, 45.0, 50.0])

    t = gtr.Translation(new_orig)
    t_inv = gtr.Translation(new_orig * -1)

    R = gtr._rodrigues_to_dcm(TEST_UVEC, np.pi)

    # change origin, rotate 180
    p1 = gtr.PivotRotation(R, new_orig)(points)

    # do the steps manually
    p2 = t_inv(points)
    p2 = gtr.Rotation(R)(p2)
    p2 = t(p2)

    assert np.all(p1 == p2)


def _check_morphology_translate(m_a, m_b, t):
    # soma points
    assert np.allclose((m_b.soma.points[:, COLS.XYZ] - m_a.soma.points[:, COLS.XYZ]), t)
    _check_neurite_translate(m_a.neurites, m_b.neurites, t)


def _check_neurite_translate(nrts_a, nrts_b, t):
    # neurite sections
    for sa, sb in zip(iter_sections(nrts_a), iter_sections(nrts_b)):
        assert np.allclose((sb.points[:, COLS.XYZ] - sa.points[:, COLS.XYZ]), t)


def test_translate_morphology_swc():
    t = np.array([100.0, 100.0, 100.0])
    m = load_morphology(SWC_NRN_PATH)
    tm = gtr.translate(m, t)
    _check_morphology_translate(m, tm, t)


def test_transform_translate_morphology_swc():
    t = np.array([100.0, 100.0, 100.0])
    m = load_morphology(SWC_NRN_PATH)
    tm = m.transform(gtr.Translation(t))
    _check_morphology_translate(m, tm, t)


def test_translate_morphology_h5():
    t = np.array([100.0, 100.0, 100.0])
    m = load_morphology(H5_NRN_PATH)
    tm = gtr.translate(m, t)

    _check_morphology_translate(m, tm, t)


def test_transform_translate_morphology_h5():
    t = np.array([100.0, 100.0, 100.0])
    m = load_morphology(H5_NRN_PATH)
    tm = m.transform(gtr.Translation(t))
    _check_morphology_translate(m, tm, t)


def test_transform__mut_immut():
    t = np.array([100.0, 100.0, 100.0])

    morph = morphio.Morphology(H5_NRN_PATH)

    m1 = load_morphology(morph)
    m2 = m1.transform(gtr.Translation(t))

    assert isinstance(m2.to_morphio(), morphio.Morphology), type(m2.to_morphio())

    _check_morphology_translate(m1, m2, t)

    morph = morphio.mut.Morphology(H5_NRN_PATH)

    m3 = load_morphology(morph)
    m4 = m3.transform(gtr.Translation(t))

    assert isinstance(m4.to_morphio(), morphio.mut.Morphology), type(m4.to_morphio())

    _check_morphology_translate(m3, m4, t)


def _apply_rot(points, rot_mat):
    return np.dot(rot_mat, np.array(points).T).T


def _check_morphology_rotate(m_a, m_b, rot_mat):
    # soma points
    assert np.allclose(
        _apply_rot(m_a.soma.points[:, COLS.XYZ], rot_mat), m_b.soma.points[:, COLS.XYZ]
    )

    # neurite sections
    _check_neurite_rotate(m_a.neurites, m_b.neurites, rot_mat)


def _check_neurite_rotate(nrt_a, nrt_b, rot_mat):
    for sa, sb in zip(iter_sections(nrt_a), iter_sections(nrt_b)):
        assert np.allclose(sb.points[:, COLS.XYZ], _apply_rot(sa.points[:, COLS.XYZ], rot_mat))


def test_rotate_morphology_swc():
    m_a = load_morphology(SWC_NRN_PATH)
    m_b = gtr.rotate(m_a, [0, 0, 1], math.pi / 2.0)
    rot = gtr._rodrigues_to_dcm([0, 0, 1], math.pi / 2.0)
    _check_morphology_rotate(m_a, m_b, rot)


def test_transform_rotate_morphology_swc():
    rot = gtr.Rotation(ROT_90)
    m_a = load_morphology(SWC_NRN_PATH)
    m_b = m_a.transform(rot)
    _check_morphology_rotate(m_a, m_b, ROT_90)


def test_rotate_morphology_h5():
    m_a = load_morphology(H5_NRN_PATH)
    m_b = gtr.rotate(m_a, [0, 0, 1], math.pi / 2.0)
    rot = gtr._rodrigues_to_dcm([0, 0, 1], math.pi / 2.0)
    _check_morphology_rotate(m_a, m_b, rot)


def test_transform_rotate_morphology_h5():
    rot = gtr.Rotation(ROT_90)
    m_a = load_morphology(H5_NRN_PATH)
    m_b = m_a.transform(rot)
    _check_morphology_rotate(m_a, m_b, ROT_90)


def test_rodrigues_to_dcm():
    RES = np.array(
        [
            [0.50017235, -0.80049871, 0.33019604],
            [0.80739289, 0.56894174, 0.15627544],
            [-0.3129606, 0.18843328, 0.9308859],
        ]
    )

    R = gtr._rodrigues_to_dcm(TEST_UVEC, TEST_ANGLE)

    # assess rotation matrix properties:

    # detR = +=1
    assert_almost_equal(np.linalg.det(R), 1.0)

    # R.T = R^-1
    assert np.allclose(np.linalg.inv(R), R.transpose())

    # check against calculated matrix
    assert np.allclose(R, RES)

    # check if opposite sign generates inverse
    Rinv = gtr._rodrigues_to_dcm(TEST_UVEC, -TEST_ANGLE)

    assert np.allclose(np.dot(Rinv, R), np.identity(3))

    # check basic rotations with a range of angles
    for angle in np.linspace(0.0, 2.0 * np.pi, 10):
        Rx = gtr._rodrigues_to_dcm(np.array([1.0, 0.0, 0.0]), angle)
        Ry = gtr._rodrigues_to_dcm(np.array([0.0, 1.0, 0.0]), angle)
        Rz = gtr._rodrigues_to_dcm(np.array([0.0, 0.0, 1.0]), angle)

        assert np.allclose(Rx, _Rx(angle))
        assert np.allclose(Ry, _Ry(angle))
        assert np.allclose(Rz, _Rz(angle))
