# Copyright (c) 2015, Ecole Polytechnique Federal de Lausanne, Blue Brain Project
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
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
