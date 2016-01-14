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

''' Module for morphology data loading and access

Data is unpacked into a 2-dimensional raw data block:

    [X, Y, Z, R, TYPE, ID, PARENT_ID]

There is one such row per measured point.

Functions to umpack the data and a higher level wrapper are provided. See

* load_data
* RawDataWrapper
'''
import os
from collections import defaultdict
from itertools import ifilter

import numpy as np

from neurom.core.point import as_point
from neurom.core.dataformat import COLS
from neurom.core.dataformat import ROOT_ID
from neurom.utils import memoize


def load_data(filename):
    '''Unpack filename and return a RawDataWrapper object containing the data

    Determines format from extension. Currently supported:

        * SWC (case-insensitive extension ".swc")
        * H5 v1 and v2 (case-insensitive extension ".h5"). Attempts to
          determine the version from the contents of the file.
    '''
    def unpack_data():
        '''Read an SWC file and return a tuple of data, offset, format.
        Forwards filename to appropriate reader depending on extension'''
        @memoize
        def _h5_reader():
            '''Lazy loading of HDF5 reader'''
            from .hdf5 import H5
            return H5.read

        @memoize
        def _swc_reader():
            '''Lazy loading of SWC reader'''
            from .swc import SWC
            return SWC.read

        _READERS = {'swc': _swc_reader, 'h5': _h5_reader}
        extension = os.path.splitext(filename)[1][1:]
        return _READERS[extension.lower()]()(filename)

    return RawDataWrapper(unpack_data())


class RawDataWrapper(object):
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
        self._id_map = {}
        for i, row in enumerate(self.data_block):
            self.adj_list[int(row[COLS.P])].append(int(row[COLS.ID]))
            self._id_map[int(row[COLS.ID])] = i
        self._ids = np.array(self.data_block[:, COLS.ID],
                             dtype=np.int32).tolist()
        self._id_set = set(self._ids)

    def get_children(self, idx):
        ''' get list of ids of children of parent with id idx'''
        if idx in self._id_set or idx == ROOT_ID:
            return self.adj_list[idx]

        raise LookupError('Invalid id: {0}'.format(idx))

    def _get_idx(self, idx):
        ''' Apply global offset to an id'''
        return self._id_map[idx]

    def get_parent(self, idx):
        '''get the parent of element with id idx'''
        if idx not in self._id_set:
            raise LookupError('Invalid id: {0}'.format(idx))
        return int(self.data_block[self._get_idx(idx)][COLS.P])

    def get_point(self, idx):
        '''Get point data for element idx'''
        idx = self._get_idx(idx)
        p = as_point(self.data_block[idx]) if idx > ROOT_ID else None
        return p

    def get_row(self, idx):
        '''Get row from idx'''
        idx = self._get_idx(idx)
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
        if pred is None:
            return list(self._ids)
        else:
            return list(r[COLS.ID] for r in self.iter_row(None, pred))

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
