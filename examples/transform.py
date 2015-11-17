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

''' Tranformations'''

import numpy as np
from neurom.core.tree import make_copy, ipreorder, val_iter
from neurom.core.dataformat import COLS

def _affineTransform(T, tree, origin=np.array([0.,0.,0.])):

    # the 3x3 Rotation matrix in the
    # 4x4 affine transform matrix T 
    R = T[0:3,0:3]
    t = T[0:3, 3]

    res_tree = make_copy(tree)

    for v in val_iter(ipreorder(res_tree)):

        v[:COLS.R] = np.dot(R, v[:COLS.R]) + t

    return res_tree

def translate(tree, t):

    T = np.zeros([4,4])

    # no rotation -> identity matrix
    T[0:3, 0:3] = np.identity(3)

    # translation
    T[0:3, 3] = t

    return _affineTransform(T, tree)

def rodriguezToRotationMatrix(axis, angle):

    ux, uy, uz = axis

    R = np.zeros([3,3])
    
    uxx = ux * ux
    uyy = uy * uy
    uzz = uz * uz
    uxy = ux * uy
    uxz = ux * uz
    uyz = uy * uz

    sn = np.sin(angle)
    cs = np.cos(angle)
    cs1 = 1. - cs

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


def rotate(tree, axis, angle, origin=np.array([0., 0., 0.])):

    T = np.zeros([4,4])

    T[0:3,0:3] = rodriguezToRotationMatrix(axis, angle)

    return _affineTransform(T, tree)


def debug_rodriguezToRotationMatrix(axis, angle):
        ''' Rodriguez formula
        '''
        I3 = np.identity(3)
        cs = np.cos(angle)

        R = cs * I3 +  np.sin(angle) * np.cross(I3, axis) + (1. - cs) * np.outer(axis, axis)

        return R
