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

'''Transformation functions for tree objects'''

import numpy as np
from neurom.core.tree import make_copy, ipreorder, val_iter
from neurom.core.dataformat import COLS


def translate(tree, t):
    '''
    Translate the tree by t

    Input :

        tree : tree object
        t : 3x1 translation vector

    Returns:

        A copy of the tree with the applied translation.
    '''
    # no rotation -> identity matrix
    return _affineTransform(tree, np.identity(3), t)


def rotate(tree, axis, angle, origin=np.zeros(3)):
    '''
    Rotation around unit vector following the right hand rule

    Input:

        tree : object tree
        axis : unit vector for the axis of rotation
        angle : rotation angle in rads

    Returns:

        A copy of the tree with the applied translation.
    '''

    R = _rodriguesToRotationMatrix(axis, angle)
    return _affineTransform(tree, R, np.zeros(3), origin=origin)


def _sin(x):
    '''sine with case for pi'''
    return 0. if np.isclose(np.mod(x, np.pi), 0.) else np.sin(x)


def _rodriguesToRotationMatrix(axis, angle):
    '''
    Generates transformation matrix from unit vector
    and rotation angle. The rotation is applied in the direction
    of the axis which is a unit vector following the right hand rule.

    Inputs :

        axis : unit vector of the direction of the rotation
        angle : angle of rotation in rads

    Returns : 3x3 Rotation matrix
    '''
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


def _affineTransform(tree, A, t, origin=None):
    '''
    Apply an affine transform on your tree by applying a linear
    transform A (e.g. rotation) and a non-linear transform t (translation)

    Input:

        A : 3x3 transformation matrix
        t : 3x1 translation array
        tree : tree object
        origin : the point with respect of which the rotation is applied. If
                 None then the x,y,z of the root node is assumed to be the
                 origin.

    Returns:

        A copy of the tree with the affine transform. Original tree is left
        unchanged
    '''
    res_tree = make_copy(tree)

    # if no origin is specified, the position from the root node
    # becomes the origin
    if origin is None:

        origin = res_tree.value[:COLS.R]

    for v in val_iter(ipreorder(res_tree)):

        v[:COLS.R] = np.dot(A, v[:COLS.R] - origin) + t + origin

    return res_tree
