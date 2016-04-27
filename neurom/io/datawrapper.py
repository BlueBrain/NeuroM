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
from itertools import ifilter
from collections import defaultdict
from ..core.dataformat import COLS
from ..core.point import as_point
from ..core.dataformat import POINT_TYPE


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
    def __init__(self, raw_data, fmt, sections=None):
        self.data_block = raw_data
        self.fmt = fmt
        self.sections = sections
        self.adj_list = defaultdict(list)

        # this loop takes all the time in the world
        for row in self.data_block:
            self.adj_list[int(row[COLS.P])].append(int(row[COLS.ID]))

    def get_children(self, idx):
        ''' get list of ids of children of parent with id idx

        Returns empty list if no parent with id idx
        '''
        return self.adj_list[idx]

    def get_parent(self, idx):
        '''get the parent of element with id idx'''
        return int(self.data_block[idx][COLS.P])

    def get_point(self, idx):
        '''Get point data for element idx'''
        return as_point(self.data_block[idx])

    def get_row(self, idx):
        '''Get row from idx'''
        return self.data_block[idx]

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
