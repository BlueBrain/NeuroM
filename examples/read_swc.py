''' Simple example for reading SWC file and accessing its data '''
from collections import defaultdict
import numpy as np

# SWC n, T, x, y, z, R, P
(ID, TYPE, X, Y, Z, R, P) = xrange(7)


class SWCData(object):
    ''' SWC loader '''
    def __init__(self, filename):
        self.data_block = np.loadtxt(filename)
        self.adj_list = defaultdict(list)
        self.offset_ = self.data_block[0][0]
        for row in self.data_block:
            self.adj_list[int(row[P])].append(int(row[ID]))

    def get_children(self, idx):
        ''' get list of ids of children of parent with id idx'''
        return self.adj_list[idx]

    def _apply_offset(self, idx):
        ''' Apply global offset to an id'''
        return idx - self.offset_

    def get_parent(self, idx):
        '''get the parent of element with id idx'''
        return int(self.data_block[self._apply_offset(idx)][P])

    def get_data(self, idx):
        '''Get row of data for element idx'''
        new_idx = self._apply_offset(idx)
        return self.data_block[new_idx] if new_idx > -1 else None

    def get_end_points(self):
        ''' get the end points of the tree

        End points have no children so are not in the
        adjacency list.
        '''
        return set(self.data_block[:, ID]) - set(self.adj_list.keys())

    def get_fork_points(self):
        '''Get list of point ids for points with more than one child'''
        return [i for i, l in self.adj_list.iteritems() if len(l) > 1]

    def traverse_unordered(self, function):
        '''Traverse the tree and apply a function to each node'''
        for row in self.data_block:
            function(row)

    def traverse_postorder(self, function, idx=-1):
        '''Traverse the tree post-order and apply a function to each node'''
        if idx not in self.data_block[:, ID] and idx != -1:
            return

        for c in self.get_children(idx):
            self.traverse_postorder(function, c)

        function(self.get_data(idx))

    def traverse_preorder(self, function, idx=-1):
        '''Traverse the tree pre-order and apply a function to each node'''
        if idx not in self.data_block[:, ID] and idx != -1:
            return

        function(self.get_data(idx))

        for c in self.get_children(idx):
            self.traverse_preorder(function, c)
