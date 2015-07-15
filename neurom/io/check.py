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
from neurom.core.dataformat import ROOT_ID
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE


def has_sequential_ids(raw_data):
    '''Check that IDs are increasing and consecutive

    returns tuple (bool, list of IDs that are not consecutive
    with their predecessor)
    '''
    ids = raw_data.get_col(COLS.ID)
    steps = [int(j) for (i, j) in zip(ids, ids[1:]) if int(j - i) != 1]
    return len(steps) == 0, steps


def has_soma_points(raw_data):
    '''Checks if the TYPE column of raw data block has
    an element of type soma'''
    return POINT_TYPE.SOMA in raw_data.get_col(COLS.TYPE)


def has_all_finite_radius_neurites(raw_data):
    '''Check that all points with neurite type have a finite radius

    return: tuple of (bool, [IDs of neurite points with zero radius])
    '''
    bad_pts = [int(row[COLS.ID]) for row in raw_data.iter_row()
               if row[COLS.TYPE] in POINT_TYPE.NEURITES and row[COLS.R] == 0.0]
    return len(bad_pts) == 0, bad_pts


def is_neurite_segment(segment):
    '''Check that both points in a segment are of neurite type

    argument:
        segment: pair of raw data rows representing a segment

    return: true if both have neurite type
    '''
    return (segment[0][COLS.TYPE] in POINT_TYPE.NEURITES and
            segment[1][COLS.TYPE] in POINT_TYPE.NEURITES)


def has_all_finite_length_segments(raw_data):
    '''Check that all segments of neurite type have a finite length

    return: tuple of (bool, [(pid, id) for zero-length segments])
    '''
    bad_segments = list()
    for row in raw_data.iter_row():
        idx = int(row[COLS.ID])
        pid = int(row[COLS.P])
        if pid != ROOT_ID:
            prow = raw_data.get_row(pid)
            if (is_neurite_segment((prow, row)) and
                    np.all(row[COLS.X: COLS.R] == prow[COLS.X: COLS.R])):
                bad_segments.append((pid, idx))
    return len(bad_segments) == 0, bad_segments
