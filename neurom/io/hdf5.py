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

''' Module for morphology HDF5 data loading

Data is unpacked into a 2-dimensional raw data block:

    [X, Y, Z, R, TYPE, ID, PARENT_ID]


HDF5.V1 Input row format:
            points: [X, Y, Z, D] (ID is position)
            groups: [FIRST_POINT_ID, TYPE, PARENT_GROUP_ID]

There is one such row per measured point.

'''
from collections import namedtuple
from itertools import izip_longest
import h5py
import numpy as np
from ..core.dataformat import COLS
from .datawrapper import RawDataWrapper


def get_version(h5file):
    '''Determine whether an HDF5 file is v1 or v2

    Return: 'H5V1', 'H5V2' or None
    '''
    if 'points' in h5file and 'structure' in h5file:
        return 'H5V1'
    elif 'neuron1/structure' in h5file:
        return 'H5V2'


class H5(object):
    '''Read HDF5 v1 or v2 files and unpack into internal raw data block

    Internal row format: [X, Y, Z, R, TYPE, ID, PARENT_ID]
    '''

    (PX, PY, PZ, PD) = xrange(4)  # points
    (GPFIRST, GTYPE, GPID) = xrange(3)  # groups or structure

    Section = namedtuple('Section', 'ids, ntype, pid')

    @staticmethod
    def read(filename, remove_duplicates=True, wrapper=RawDataWrapper):
        '''Read a file and return a tuple of data, format.

        * Tries to guess the format and the H5 version.
        * Unpacks the first block it finds out of ('repaired', 'unraveled', 'raw')

        Parameters:
            remove_duplicates: boolean, \
            If True removes duplicate points \
            from the beginning of each section.
        '''
        h5file = h5py.File(filename, mode='r')
        version = get_version(h5file)
        if version == 'H5V1':
            points, groups = _unpack_v1(h5file)
        elif version == 'H5V2':
            stg = next(s for s in ('repaired', 'unraveled', 'raw')
                       if s in h5file['neuron1'])
            points, groups = _unpack_v2(h5file, stage=stg)

        h5file.close()

        data, sec = H5.unpack_data(points, groups, remove_duplicates)

        return wrapper(data, version, sec)

    @staticmethod
    def unpack_data(points, groups, remove_duplicates):
        '''Unpack data from h5 data groups into internal format'''

        if remove_duplicates:
            points, groups = H5.remove_duplicate_points(points, groups)

        n_points = len(points)
        group_ids = groups[:, H5.GPFIRST]

        # point_id -> group_id map
        pid_map = np.zeros(n_points)
        #  point ID -> type map
        typ_map = np.zeros(n_points)
        # sections (ids, type, parent_id)
        sections = [0] * len(group_ids)

        for i, (j, k) in enumerate(izip_longest(group_ids,
                                                group_ids[1:],
                                                fillvalue=n_points)):
            j = int(j)
            k = int(k)
            sections[i] = H5.Section(slice(j, k), groups[i][H5.GTYPE], groups[i][H5.GPID])
            typ_map[j: k] = groups[i][H5.GTYPE]
            # parent is last point in previous group
            pid_map[j] = group_ids[groups[i][H5.GPID] + 1] - 1
            # parent is previous point
            pid_map[j + 1: k] = np.arange(j, k - 1)

        db = np.zeros((n_points, 7))
        db[:, : H5.PD + 1] = points
        db[:, H5.PD] /= 2  # Store radius, not diameter
        db[:, COLS.ID] = np.arange(n_points)
        db[:, COLS.P] = pid_map
        db[:, COLS.TYPE] = typ_map

        return db, sections

    @staticmethod
    def remove_duplicate_points(points, groups):
        ''' Removes the duplicate points from the beginning of a section,
        if they are present in points-groups representation.

        Returns:
            points, groups with unique points.

        '''

        group_initial_ids = groups[:, H5.GPFIRST]

        to_be_reduced = np.zeros(len(group_initial_ids))
        to_be_removed = []

        for ig, g in enumerate(groups):
            iid, typ, pid = g[H5.GPFIRST], g[H5.GTYPE], g[H5.GPID]
            # Remove first point from sections that are
            # not the root section, a soma, or a child of a soma
            if pid != -1 and typ != 1 and groups[pid][H5.GTYPE] != 1:
                # Remove duplicate from list of points
                to_be_removed.append(iid)
                # Reduce the id of the following sections
                # in groups structure by one
                to_be_reduced[ig + 1:] += 1

        groups[:, H5.GPFIRST] = groups[:, H5.GPFIRST] - to_be_reduced
        points = np.delete(points, to_be_removed, axis=0)

        return points, groups


def _unpack_v1(h5file):
    '''Unpack groups from HDF5 v1 file'''
    points = np.array(h5file['points'])
    groups = np.array(h5file['structure'])
    return points, groups


def _unpack_v2(h5file, stage):
    '''Unpack groups from HDF5 v2 file'''
    points = np.array(h5file['neuron1/%s/points' % stage])
    # from documentation: The /neuron1/structure/unraveled reuses /neuron1/structure/raw
    groups_stage = stage if stage != 'unraveled' else 'raw'
    groups = np.array(h5file['neuron1/structure/%s' % groups_stage])
    stypes = np.array(h5file['neuron1/structure/sectiontype'])
    groups = np.hstack([groups, stypes])
    groups[:, [1, 2]] = groups[:, [2, 1]]
    return points, groups
