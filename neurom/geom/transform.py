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

"""Transformation functions for morphology objects."""

import numpy as np


_TRANSFDOC = """

    The transformation can be applied to [x, y, z] points via a call operator
    with the following properties:

    Parameters:
        points: 2D numpy array of points, where the 3 columns
            are the x, y, z coordinates respectively

    Returns:
        2D numpy array of transformed points
"""


class Transform3D(object):
    """Class representing a generic 3D transformation."""
    __doc__ += _TRANSFDOC

    def __call__(self, points):
        """Apply a 3D transformation to a set of points."""
        raise NotImplementedError


class Translation(Transform3D):
    """Class representing a 3D translation."""
    __doc__ += _TRANSFDOC

    def __init__(self, translation):
        """Initialize a 3D translation.

        Arguments:
            translation: 3-vector of x, y, z
        """
        self._trans = np.array(translation)

    def __call__(self, points):
        """Apply a 3D translation to a set of points."""
        return points + self._trans


class Rotation(Transform3D):
    """Class representing a 3D rotation."""
    __doc__ += _TRANSFDOC

    def __init__(self, dcm):
        """Initialize a 3D rotation.

        Arguments:
            dcm: a 3x3 direction cosine matrix
        """
        self._dcm = np.array(dcm)

    def __call__(self, points):
        """Apply a 3D rotation to a set of points."""
        return np.dot(self._dcm, np.array(points).T).T


class PivotRotation(Rotation):
    """Class representing a 3D rotation about a pivot point."""
    __doc__ += _TRANSFDOC

    def __init__(self, dcm, pivot=None):
        """Initialize a 3D rotation about a pivot point.

        Arguments:
            dcm: a 3x3 direction cosine matrix
            pivot: a 3-vector specifying the origin of rotation
        """
        super().__init__(dcm)
        self._origin = np.zeros(3) if pivot is None else np.array(pivot)

    def __call__(self, points):
        """Apply a 3D pivoted rotation to a set of points."""
        points = points - self._origin
        points = np.dot(self._dcm, np.array(points).T).T
        points += self._origin
        return points


def translate(obj, t):
    """Translate object of supported type.

    Arguments:
        obj : object to be translated. Must implement a transform method.
        t: translation 3-vector

    Returns:
        copy of the object with the applied translation
    """
    try:
        return obj.transform(Translation(t))
    except AttributeError as e:
        raise NotImplementedError from e


def rotate(obj, axis, angle, origin=None):
    """Rotation around unit vector following the right hand rule.

    Arguments:
        obj : obj to be rotated (e.g. neurite, neuron).
            Must implement a transform method.
        axis : unit vector for the axis of rotation
        angle : rotation angle in rads
        origin : specify the origin about which rotation occurs

    Returns:
        A copy of the object with the applied translation.
    """
    R = _rodrigues_to_dcm(axis, angle)

    try:
        return obj.transform(PivotRotation(R, origin))
    except AttributeError as e:
        raise NotImplementedError from e


def _sin(x):
    """Sine with case for pi multiples."""
    return 0. if np.isclose(np.mod(x, np.pi), 0.) else np.sin(x)


def _rodrigues_to_dcm(axis, angle):
    """Generates transformation matrix from unit vector and rotation angle.

    The rotation is applied in the direction
    of the axis which is a unit vector following the right hand rule.

    Inputs :

        axis : unit vector of the direction of the rotation
        angle : angle of rotation in rads

    Returns : 3x3 Rotation matrix
    """
    ux, uy, uz = axis / np.linalg.norm(axis)

    uxx = ux * ux
    uyy = uy * uy
    uzz = uz * uz
    uxy = ux * uy
    uxz = ux * uz
    uyz = uy * uz

    sn = _sin(angle)
    cs = _sin(np.pi / 2. - angle)
    cs1 = 1. - cs

    R = np.zeros([3, 3])

    R[0, 0] = cs + uxx * cs1
    R[0, 1] = uxy * cs1 - uz * sn
    R[0, 2] = uxz * cs1 + uy * sn

    R[1, 0] = uxy * cs1 + uz * sn
    R[1, 1] = cs + uyy * cs1
    R[1, 2] = uyz * cs1 - ux * sn

    R[2, 0] = uxz * cs1 - uy * sn
    R[2, 1] = uyz * cs1 + ux * sn
    R[2, 2] = cs + uzz * cs1

    return R
