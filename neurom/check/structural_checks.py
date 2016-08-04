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
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE
from neurom.core import make_soma
from neurom.fst._core import make_neurites
from neurom.exceptions import SomaError


def has_sequential_ids(data_wrapper):
    '''Check that IDs are increasing and consecutive

    returns tuple (bool, list of IDs that are not consecutive
    with their predecessor)
    '''
    db = data_wrapper.data_block
    ids = db[:, COLS.ID]
    steps = ids[np.where(np.diff(ids) != 1)[0] + 1].astype(int)
    return CheckResult(len(steps) == 0, steps)


def no_missing_parents(data_wrapper):
    '''Check that all points have existing parents
    Point's parent ID must exist and parent must be declared
    before child.

    Returns:
        tuple (bool, list of IDs that have no parent)
    '''
    db = data_wrapper.data_block
    ids = np.setdiff1d(db[:, COLS.P], db[:, COLS.ID])[1:]
    return CheckResult(len(ids) == 0, ids.astype(np.int) + 1)


def is_single_tree(data_wrapper):
    '''Check that data forms a single tree

    Only the first point has ID of -1.

    Note:
        This assumes no_missing_parents passed.
    '''
    db = data_wrapper.data_block
    bad_ids = db[db[:, COLS.P] == -1][1:, COLS.ID]
    return CheckResult(len(bad_ids) == 0, bad_ids.tolist())


def has_increasing_ids(data_wrapper):
    '''Check that IDs are increasing

    returns tuple (bool, list of IDs that are inconsistent
    with their predecessor)
    '''
    db = data_wrapper.data_block
    ids = db[:, COLS.ID]
    steps = ids[np.where(np.diff(ids) <= 0)[0] + 1].astype(int)
    return CheckResult(len(steps) == 0, steps)


def has_soma_points(data_wrapper):
    '''Checks if the TYPE column of raw data block has
    an element of type soma'''
    db = data_wrapper.data_block
    return CheckResult(POINT_TYPE.SOMA in db[:, COLS.TYPE], None)


def has_all_finite_radius_neurites(data_wrapper, threshold=0.0):
    '''Check that all points with neurite type have a finite radius

    return: tuple of (bool, [IDs of neurite points with zero radius])
    '''
    db = data_wrapper.data_block
    neurite_ids = np.in1d(db[:, COLS.TYPE], POINT_TYPE.NEURITES)
    zero_radius_ids = db[:, COLS.R] <= threshold
    bad_pts = np.array(db[neurite_ids & zero_radius_ids][:, COLS.ID],
                       dtype=int).tolist()
    return CheckResult(len(bad_pts) == 0, bad_pts)


def has_valid_soma(data_wrapper):
    '''Check if a data block has a valid soma'''
    try:
        make_soma(data_wrapper.soma_points())
        return CheckResult(True)
    except SomaError:
        return CheckResult(False)


def has_valid_neurites(data_wrapper):
    '''Check if any neurites can be reconstructed from data block'''
    n, _ = make_neurites(data_wrapper)
    return CheckResult(len(n) > 0)
