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
from bisect import bisect_right

import h5py
import numpy as np


def get_version(h5file):
    '''Determine whether an HDF5 file is v1 or v2

    Return: 'H5V1', 'H5V2' or None
    '''
    if 'points' in h5file and 'structure' in h5file:
        return 'H5V1'
    elif 'neuron1/structure' in h5file:
        return 'H5V2'


class H5V1(object):
    '''Read HDF5 v1 files and unpack into internal raw data block

    Input row format:
        points: (PX, PY, PZ, PD) -> [X, Y, Z, D] (ID is position)
        groups: (GPFIRST, GTYPE, GPID) -> [FIRST_POINT_ID, TYPE, PARENT_GROUP_ID]

    Internal row format: [X, Y, Z, R, TYPE, ID, PARENT_ID]
    '''

    (PX, PY, PZ, PD) = xrange(4)  # points
    (GPFIRST, GTYPE, GPID) = xrange(3)  # groups or structure


class H5(object):
    '''Read HDF5 v1 or v2 files and unpack into internal raw data block

    Internal row format: [X, Y, Z, R, TYPE, ID, PARENT_ID]
    '''

    @staticmethod
    def read_v1(filename):
        '''Read an HDF5 v1 file and return a tuple of data, offset, format.'''
        points, groups = _unpack_v1(h5py.File(filename))
        data = H5.unpack_data(points, groups)
        offset = 0  # H5 is index based, so there's no offset
        return data, offset, 'H5V1'

    @staticmethod
    def read_v2(filename, stage='raw'):
        '''Read an HDF5 v2 file and return a tuple of data, offset, format.'''
        points, groups = _unpack_v2(h5py.File(filename), stage)
        data = H5.unpack_data(points, groups)
        offset = 0  # H5 is index based, so there's no offset
        return data, offset, 'H5V2'

    @staticmethod
    def read(filename):
        '''Read a file and return a tuple of data, offset, format.

        * Tries to guess the format and the H5 version.
        * Unpacks the first block it finds out of ('repaired', 'unraveled', 'raw')
        '''
        h5file = h5py.File(filename)
        version = get_version(h5file)
        if version == 'H5V1':
            points, groups = _unpack_v1(h5py.File(filename))
        elif version == 'H5V2':
            stg = next(s for s in ('repaired', 'unraveled', 'raw')
                       if s in h5file['neuron1'])
            points, groups = _unpack_v2(h5py.File(filename), stage=stg)
        data = H5.unpack_data(points, groups)
        offset = 0  # H5 is index based, so there's no offset
        return data, offset, version

    @staticmethod
    def unpack_data(points, groups):
        '''Unpack data from h5 data groups into internal format'''

        def find_group(point_id):
            '''Find the structure group a points id belongs to

            Return: group or section point_id belongs to. Last group if
                    point_id out of bounds.
            '''
            bs = bisect_right(groups[:, H5V1.GPFIRST], point_id)
            bs = max(bs - 1, 0)
            return groups[bs]

        def find_parent_id(point_id):
            '''Find the parent ID of a point'''
            group = find_group(point_id)
            if point_id != group[H5V1.GPFIRST]:
                # point is not first point in section
                # so parent is previous point
                return point_id - 1
            else:
                # parent is last point in parent group
                parent_group_id = group[H5V1.GPID]
                # get last point in parent group
                return groups[parent_group_id + 1][H5V1.GPFIRST] - 1

        return np.array([(p[H5V1.PX], p[H5V1.PY], p[H5V1.PZ], p[H5V1.PD] / 2.,
                          find_group(i)[H5V1.GTYPE], i,
                          find_parent_id(i))
                         for i, p in enumerate(points)])


def _unpack_v1(h5file):
    '''Unpack groups from HDF5 v1 file'''
    points = np.array(h5file['points'])
    groups = np.array(h5file['structure'])
    return points, groups


def _unpack_v2(h5file, stage):
    '''Unpack groups from HDF5 v2 file'''
    points = np.array(h5file['neuron1/%s/points' % stage])
    groups = np.array(h5file['neuron1/structure/%s' % stage])
    stypes = np.array(h5file['neuron1/structure/sectiontype'])
    groups = np.hstack([groups, stypes])
    groups[:, [1, 2]] = groups[:, [2, 1]]
    return points, groups
