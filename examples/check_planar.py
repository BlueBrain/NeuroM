#!/usr/bin/env python
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

'''Function that checks if tree is planar'''


import numpy as np
from neurom.core.tree import val_iter, ipreorder
from neurom.core.dataformat import COLS


def pca(points):
    '''
    Estimate the principal components of the given point cloud

    Input
        A numpy array of points of the form ((x1,y1,z1), (x2, y2, z2)...)

    Ouptut
        Eigenvalues and respective eigenvectors
    '''

    # calculate the covariance of the points
    cov = np.cov(points)

    # find the principal components
    return np.linalg.eig(cov)


def get_extent(tree):
    '''Calculate the extent of a tree, that is the maximum distance between
        the projections on the principal directions of the covariance matrix
        of the x,y,z points of the nodes of the tree.

        Input
            tree : a tree object

        Returns

            extents : the extents for each of the eigenvectors of the cov matrix
            eigs : eigenvalues of the covariance matrix
            eigv : respective eigenvectors of the covariance matrix
    '''
    # extract the x,y,z coordinates of all the points in the tree
    points = np.array([value[COLS.X: COLS.R]for value in val_iter(ipreorder(tree))])

    # center the points around 0.0
    points -= np.mean(points, axis=0)

    # principal components
    eigs, eigv = pca(points.transpose())

    extent = np.zeros(3)

    for i, v in enumerate(eigv):

        # orthogonal projection onto the direction of the v component
        scalar_projs = np.sort(np.array([np.dot(p, v) for p in points]))

        extent[i] = scalar_projs[-1]

        if scalar_projs[0] < 0.:
            extent -= scalar_projs[0]

    return extent, eigs, eigv


def is_flat(tree, tol, method='tolerance'):
    '''Check if neurite is flat using the given method

        Input

            tree : the tree object

            tol : the tolerance

            method : the method of flatness estimation.
            'tolerance' returns true if any extent of the tree
            is smaller than the given tolerance
            'ratio' returns true if the ratio of the smallest directions
            is smaller than tol. e.g. [1,2,3] -> 1/2 < tol

        Returns

            True if it is flat

    '''

    ext, _, _ = get_extent(tree)

    # check if the ratio between the two smaller directions
    # is smaller than the tol ratio
    if method == 'ratio':

        sorted_ext = np.sort(ext)
        return sorted_ext[0] / sorted_ext[1] < float(tol)

    # check if any of the extents is smaller than the tolerance
    else:

        return any(ext < float(tol))
