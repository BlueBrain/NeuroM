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
import h5py
import numpy as np
from itertools import izip_longest, ifilter
from collections import defaultdict
from ..core.dataformat import COLS
from ..core.point import as_point
from ..core.dataformat import ROOT_ID
from ..core.dataformat import POINT_TYPE


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

        return _RDW((data, version))

    @staticmethod
    def unpack_data(points, groups):
        '''Unpack data from h5 data groups into internal format'''

        n_points = len(points)
        group_ids = groups[:, _H5STRUCT.GPFIRST]

        # point_id -> group_id map
        pid_map = np.zeros(n_points)
        #  point ID -> type map
        typ_map = np.zeros(n_points)

        for i, (j, k) in enumerate(izip_longest(group_ids,
                                                group_ids[1:],
                                                fillvalue=n_points)):
            j = int(j)
            k = int(k)
            typ_map[j: k] = groups[i][_H5STRUCT.GTYPE]
            # parent is last point in previous group
            pid_map[j] = group_ids[groups[i][_H5STRUCT.GPID] + 1] - 1
            # parent is previous point
            pid_map[j + 1: k] = np.arange(j, k - 1)

        db = np.zeros((n_points, 7))
        db[:, : _H5STRUCT.PD + 1] = points
        db[:, _H5STRUCT.PD] /= 2  # Store radius, not diameter
        db[:, COLS.ID] = np.arange(n_points)
        db[:, COLS.P] = pid_map
        db[:, COLS.TYPE] = typ_map

        return db

    @staticmethod
    def remove_duplicate_points(points, groups):
        ''' Removes the duplicate points from the beginning of a section,
        if they are present in points-groups representation.

        Returns:
            points, groups with unique points.

        '''

        group_initial_ids = groups[:, _H5STRUCT.GPFIRST]

        to_be_reduced = np.zeros(len(group_initial_ids))
        to_be_removed = []

        for ig, g in enumerate(groups):
            iid, typ, pid = g[_H5STRUCT.GPFIRST], g[_H5STRUCT.GTYPE], g[_H5STRUCT.GPID]
            # Remove first point from sections that are
            # not the root section, a soma, or a child of a soma
            if pid != -1 and typ != 1 and groups[pid][_H5STRUCT.GTYPE] != 1:
                # Remove duplicate from list of points
                to_be_removed.append(iid)
                # Reduce the id of the following sections
                # in groups structure by one
                to_be_reduced[ig + 1:] += 1

        groups[:, _H5STRUCT.GPFIRST] = groups[:, _H5STRUCT.GPFIRST] - to_be_reduced
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


class _RDW(object):
    '''Class holding an array of data and an offset to the first element
    and giving basic access to its elements

    The array contains rows with
        [X, Y, Z, R, TYPE, ID, PID]

    where the elements are

    * X, Y, Z: X, Y, Z coordinates of point
    * R: Radius of node at that point
    * TYPE: Type of neuronal segment.
    * ID: Identifier for a point. Non-negative, increases by one for each row.
    * PID: ID of parent point
    '''
    def __init__(self, raw_data):
        self.data_block, self.fmt = raw_data
        self.adj_list = defaultdict(list)
        # this loop takes all the time in the world
        for row in self.data_block:
            # and building this adjacency list takes most of that.
            self.adj_list[int(row[COLS.P])].append(int(row[COLS.ID]))

    def get_children(self, idx):
        ''' get list of ids of children of parent with id idx'''

        def _valid_id(idx):
            '''Check if idx is a valid id'''
            return idx == ROOT_ID or (len(self.data_block) > idx and -1 < idx)

        if _valid_id(idx):
            return self.adj_list[idx]

        raise LookupError('Invalid id: {0}'.format(idx))

    def get_parent(self, idx):
        '''get the parent of element with id idx'''
        return int(self.data_block[idx][COLS.P])

    def get_point(self, idx):
        '''Get point data for element idx'''
        p = as_point(self.data_block[idx]) if idx > ROOT_ID else None
        return p

    def get_row(self, idx):
        '''Get row from idx'''
        return self.data_block[idx] if idx > ROOT_ID else None

    def get_col(self, col_id):
        '''Get column from ID'''
        return self.data_block[:, col_id]

    def get_end_points(self):
        ''' get the end points of the tree

        End points have no children so are not in the
        adjacency list.
        '''
        return [int(i) for i in
                set(self.get_col(COLS.ID)) - set(self.adj_list.keys())]

    def get_ids(self, pred=None):
        '''Get the list of ids for rows satisfying an optional row predicate'''
        return list(r[COLS.ID] for r in self.iter_row(None, pred))

    def get_soma_rows(self):
        '''Get the IDs of all soma points'''
        db = self.data_block
        return db[db[:, COLS.TYPE] == POINT_TYPE.SOMA]

    def get_fork_points(self):
        '''Get list of point ids for points with more than one child'''
        return [i for i, l in self.adj_list.iteritems() if len(l) > 1]

    def iter_row(self, start_id=None, pred=None):
        '''Get an row iterator to a starting at start_id and satisfying a
        row predicate pred.
        '''
        if start_id is None:
            start_id = 0

        if start_id < 0 or start_id >= self.data_block.shape[0]:
            raise LookupError('Invalid id: {0}'.format(start_id))

        irow = iter(self.data_block[start_id:])
        return irow if pred is None else ifilter(pred, irow)
