''' Module for morphology data loading and access

Data is unpacked into a 2-dimensional raw data block following the SWC format:

    [ID, TYPE, X, Y, Z, R, PARENT]

There is one such row per measured point.

Functions to umpack the data and a higher level wrapper are provided. See

* load_data
* RawDataWrapper

Currently, only the SWC format is supported.
'''
import os
from collections import defaultdict
from itertools import ifilter
import numpy as np
from neurom.core.point import as_point
from neurom.core.dataformat import COLS
from neurom.core.dataformat import ROOT_ID


def read_swc(filename):
    '''Read an SWC file and return a tuple of data, offset, format.'''
    data = np.loadtxt(filename)
    offset = data[0][COLS.ID]
    return data, offset, 'SWC'

_READERS = {'swc': read_swc}


def unpack_data(filename):
    '''Read an SWC file and return a tuple of data, offset, format.
    Forwards filename to appropriate reader depending on extension'''
    extension = os.path.splitext(filename)[1][1:]
    return _READERS[extension.lower()](filename)


def load_data(filename):
    '''Unpack filename and return a RawDataWrapper object containing the data

    Determines format from extension. Currently supported:

        * SWC (case-insensitive extension ".swc")
    '''
    return RawDataWrapper(unpack_data(filename))


class RawDataWrapper(object):
    '''Class holding an array of data and an offset to the first element
    and giving basic access to its elements

    The array contains rows with
        [ID, TYPE, X, Y, Z, R, P]

    where the elements are

    * ID: Identifier for a point. Non-negative, increases by one for each row.
    * TYPE: Type of neuronal segment.
    * X, Y, Z: X, Y, Z coorsinates of point
    * R: Radius of node at that point
    * P: ID of parent point
    '''
    def __init__(self, raw_data):
        self.data_block, self._offset, self.fmt = raw_data
        self.adj_list = defaultdict(list)
        for row in self.data_block:
            self.adj_list[int(row[COLS.P])].append(int(row[COLS.ID]))

    def get_children(self, idx):
        ''' get list of ids of children of parent with id idx'''
        if idx != ROOT_ID and idx not in self.get_ids():
            raise LookupError('Invalid id: {0}'.format(idx))
        return self.adj_list[idx]

    def _apply_offset(self, idx):
        ''' Apply global offset to an id'''
        return idx - self._offset

    def get_parent(self, idx):
        '''get the parent of element with id idx'''
        if idx not in self.get_ids():
            raise LookupError('Invalid id: {0}'.format(idx))
        return int(self.data_block[self._apply_offset(idx)][COLS.P])

    def get_point(self, idx):
        '''Get point data for element idx'''
        idx = self._apply_offset(idx)
        p = as_point(self.data_block[idx]) if idx > ROOT_ID else None
        return p

    def get_row(self, idx):
        '''Get row from idx'''
        idx = self._apply_offset(idx)
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
        return [r[COLS.ID] for r in self.iter_row(None, pred)]

    def get_fork_points(self):
        '''Get list of point ids for points with more than one child'''
        return [i for i, l in self.adj_list.iteritems() if len(l) > 1]

    def iter_row(self, start_id=None, pred=None):
        '''Get an row iterator to a starting at start_id and satisfying a
        row predicate pred.
        '''
        if start_id is None:
            start_id = self._offset

        start_id = self._apply_offset(start_id)
        if start_id < 0 or start_id >= self.data_block.shape[0]:
            raise LookupError('Invalid id: {0}'.format(start_id))

        irow = iter(self.data_block[start_id:])
        return irow if pred is None else ifilter(pred, irow)
