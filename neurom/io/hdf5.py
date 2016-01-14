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
from neurom.utils import memoize
import h5py
import numpy as np
import itertools


def get_version(h5file):
    '''Determine whether an HDF5 file is v1 or v2

    Return: 'H5V1', 'H5V2' or None
    '''
    if 'points' in h5file and 'structure' in h5file:
        return 'H5V1'
    elif 'neuron1/structure' in h5file:
        return 'H5V2'


class _H5STRUCT(object):
    '''Define internal structure of HDF5 data

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
    def read_v1(filename, remove_duplicates=True):
        '''Read an HDF5 v1 file and return a tuple of data, format.

        Parameters:
            remove_duplicates: boolean, \
            If True removes duplicate points \
            from the beginning of each section.
        '''
        points, groups = _unpack_v1(h5py.File(filename, mode='r'))
        if remove_duplicates:
            data = H5.unpack_data(*H5.remove_duplicate_points(points, groups))
        else:
            data = H5.unpack_data(points, groups)
        return data, 'H5V1'

    @staticmethod
    def read_v2(filename, stage='raw', remove_duplicates=True):
        '''Read an HDF5 v2 file and return a tuple of data, format.

        Parameters:
            remove_duplicates: boolean, \
            If True removes duplicate points \
            from the beginning of each section.
        '''
        h5file = h5py.File(filename, mode='r')
        points, groups = _unpack_v2(h5file, stage)
        h5file.close()
        if remove_duplicates:
            data = H5.unpack_data(*H5.remove_duplicate_points(points, groups))
        else:
            data = H5.unpack_data(points, groups)
        return data, 'H5V2'

    @staticmethod
    def read(filename, remove_duplicates=True):
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

        if remove_duplicates:
            data = H5.unpack_data(*H5.remove_duplicate_points(points, groups))
        else:
            data = H5.unpack_data(points, groups)
        return data, version

    @staticmethod
    def unpack_data(points, groups):
        '''Unpack data from h5 data groups into internal format'''

        @memoize
        def find_group(point_id):
            '''Find the structure group a points id belongs to

            Return: group or section point_id belongs to. Last group if
                    point_id out of bounds.
            '''
            bs = np.searchsorted(groups[:, _H5STRUCT.GPFIRST], point_id, side='right')
            bs = max(bs - 1, 0)
            return groups[bs]

        def find_parent_id(point_id):
            '''Find the parent ID of a point'''
            group = find_group(point_id)
            if point_id != group[_H5STRUCT.GPFIRST]:
                # point is not first point in section
                # so parent is previous point
                return point_id - 1
            else:
                # parent is last point in parent group
                parent_group_id = group[_H5STRUCT.GPID]
                # get last point in parent group
                return groups[parent_group_id + 1][_H5STRUCT.GPFIRST] - 1

        return np.array([(p[_H5STRUCT.PX], p[_H5STRUCT.PY], p[_H5STRUCT.PZ], p[_H5STRUCT.PD] / 2.,
                          find_group(i)[_H5STRUCT.GTYPE], i,
                          find_parent_id(i))
                         for i, p in enumerate(points)])

    @staticmethod
    def remove_duplicate_points(points, groups):
        ''' Removes the duplicate points from the beginning of a section,
        if they are present in points-groups representation.

        Returns:
            points, groups with unique points.

        '''

        def _find_last_point(group_id, groups):
            ''' Identifies and returns the id of the last point of a group'''
            group_initial_ids = np.sort(np.transpose(groups)[0])

            if group_id != len(group_initial_ids) - 1:
                return group_initial_ids[np.where(group_initial_ids ==
                                                  groups[group_id][0])[0][0] + 1] - 1

        to_be_reduced = np.zeros(len(groups))
        to_be_removed = []

        for ig, g in enumerate(groups):
            if g[2] != -1 and np.allclose(points[g[0]],
                                          points[_find_last_point(g[2], groups)]):
                # Remove duplicate from list of points
                to_be_removed.append(g[0])
                # Reduce the id of the following sections
                # in groups structure by one
                for igg in range(ig + 1, len(groups)):
                    to_be_reduced[igg] = to_be_reduced[igg] + 1

        groups = np.array([np.subtract(i, [j, 0, 0])
                           for i, j in itertools.izip(groups, to_be_reduced)])
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
