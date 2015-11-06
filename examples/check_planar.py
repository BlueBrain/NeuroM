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


def extent_of_tree(tree):
    '''Calculate the extend of a tree, which is defined as the maximum distance
        on the direction of minimum variance.

        Input
            tree : a tree object

            tol : tolerance in microns

        Returns
            extend : int
    '''
    # extract the x,y,z coordinates of all the points in the tree
    points = np.array([value[COLS.X: COLS.R]for value in val_iter(ipreorder(tree))])

    # center the points around 0.0
    points -= np.mean(points, axis=0)

    # principal components
    eigs, eigv = pca(points.transpose())

    # smallest component size
    min_eigv = eigv[:, np.argmin(eigs)]

    # orthogonal projection onto the direction of the smallest component
    scalar_projs = np.sort(np.array([np.dot(p, min_eigv) for p in points]))

    extent = scalar_projs[-1]

    if scalar_projs[0] < 0.:
        extent -= scalar_projs[0]

    # the max distance mong the points on that direction
    return extent


def check_flat_neuron(neuron, tol=10):
    '''Iterate over neurites and check for flatness given the tolerance. default tol = 10

        Input

            neuron : neuron object

            tol : tolerance in microns
    '''

    print '\nChecking for flat neurites. Tolerance : {0} microns \n'.format(tol)

    print 'Neurite Type \t\t\t Extend \t Flat'

    print '-' * 60

    for neurite in neuron.neurites:

        extent = extent_of_tree(neurite)

        print '{0:30}   {1:.02f} \t\t {2}'.format(neurite.type, extent, extent < float(tol))
