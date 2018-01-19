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

'''Module with consistency/validity checks for raw data  blocks'''
import numpy as np
from neurom.check import CheckResult
from neurom.core.dataformat import COLS, POINT_TYPE
from neurom.exceptions import SomaError


def has_sequential_ids(neuron):
    '''Check that IDs are increasing and consecutive

    returns tuple (bool, list of IDs that are not consecutive
    with their predecessor)
    '''
    points = neuron.points
    ids = points[:, COLS.ID]
    steps = ids[np.where(np.diff(ids) != 1)[0] + 1].astype(int)
    return CheckResult(len(steps) == 0, steps)


def no_missing_parents(neuron):
    '''Check that all points have existing parents
    Point's parent ID must exist and parent must be declared
    before child.

    Returns:
        CheckResult with result and list of IDs that have no parent
    '''
    points = neuron.points
    ids = np.setdiff1d(points[:, COLS.P], points[:, COLS.ID])[1:]
    return CheckResult(len(ids) == 0, ids.astype(np.int) + 1)


def is_single_tree(neuron):
    '''Check that data forms a single tree

    Only the first point has ID of -1.

    Returns:
        CheckResult with result and list of IDs

    Note:
        This assumes no_missing_parents passed.
    '''
    points = neuron.points
    bad_ids = points[points[:, COLS.P] == -1][1:, COLS.ID]
    return CheckResult(len(bad_ids) == 0, bad_ids.tolist())


def has_increasing_ids(neuron):
    '''Check that IDs are increasing

    Returns:
        CheckResult with result and list of IDs that are inconsistent
        with their predecessor
    '''
    ids = neuron.points[:, COLS.ID]
    steps = ids[np.where(np.diff(ids) <= 0)[0] + 1].astype(int)
    return CheckResult(len(steps) == 0, steps)


def has_soma_points(neuron):
    '''Checks if the TYPE column of raw data block has an element of type soma

    Returns:
        CheckResult with result
    '''
    points = neuron.points
    return CheckResult(POINT_TYPE.SOMA in points[:, COLS.TYPE], None)


def has_all_finite_radius_neurites(neuron, threshold=0.0):
    '''Check that all points with neurite type have a finite radius

    Returns:
        CheckResult with result and list of IDs of neurite points with zero radius
    '''
    points = np.vstack([neurite.points for neurite in neuron.neurites])
    zero_radius_ids = points[:, COLS.R] <= threshold
    bad_pts = np.where(zero_radius_ids)[0].tolist()
    return CheckResult(len(bad_pts) == 0, bad_pts)
