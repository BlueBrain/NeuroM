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

'''Generic tree class and iteration functions'''
from itertools import chain, imap, ifilter
from collections import deque


class Tree(object):
    '''Simple recursive tree class'''
    def __init__(self):
        self.parent = None
        self.children = list()

    def add_child(self, tree):
        '''Add a child to the list of this tree's children

        This tree becomes the added tree's parent
        '''
        tree.parent = self
        self.children.append(tree)
        return tree

    def is_forking_point(self):
        '''Is tree a forking point?'''
        return len(self.children) > 1

    def is_bifurcation_point(self):
        '''Is tree a bifurcation point?'''
        return len(self.children) == 2

    def is_leaf(self):
        '''Is tree a leaf?'''
        return len(self.children) == 0

    def is_root(self):
        '''Is tree the root node?'''
        return self.parent is None

    def ipreorder(self):
        '''Depth-first pre-order iteration of tree nodes'''
        children = deque((self, ))
        while children:
            cur_node = children.pop()
            children.extend(reversed(cur_node.children))
            yield cur_node

    def ipostorder(self):
        '''Depth-first post-order iteration of tree nodes'''
        children = [self, ]
        seen = set()
        while children:
            cur_node = children[-1]
            if cur_node not in seen:
                seen.add(cur_node)
                children.extend(reversed(cur_node.children))
            else:
                children.pop()
                yield cur_node

    def iupstream(self):
        '''Iterate from a tree node to the root nodes'''
        t = self
        while t is not None:
            yield t
            t = t.parent

    def ileaf(self):
        '''Iterator to all leaves of a tree'''
        return ifilter(Tree.is_leaf, self.ipreorder())

    def iforking_point(self, iter_mode=ipreorder):
        '''Iterator to forking points. Returns a tree object.

        Parameters:
            tree: the tree over which to iterate
            iter_mode: iteration mode. Default: ipreorder.
        '''
        return ifilter(Tree.is_forking_point, iter_mode(self))

    def ibifurcation_point(self, iter_mode=ipreorder):
        '''Iterator to bifurcation points. Returns a tree object.

        Parameters:
            tree: the tree over which to iterate
            iter_mode: iteration mode. Default: ipreorder.
        '''
        return ifilter(Tree.is_bifurcation_point, iter_mode(self))


def i_chain2(trees, iterator_type=Tree.ipreorder, mapping=None, tree_filter=None):
    '''Returns a mapped iterator to a collection of trees

    Provides access to all the elements of all the trees
    in one iteration sequence.

    Parameters:
        trees: iterator or iterable of tree objects
        iterator_type: type of the iteration (segment, section, triplet...)
        mapping: optional function to apply to the iterator's target.
        tree_filter: optional top level filter on properties of tree objects.
    '''
    nrt = (trees if tree_filter is None
           else filter(tree_filter, trees))

    chain_it = chain.from_iterable(imap(iterator_type, nrt))
    return chain_it if mapping is None else imap(mapping, chain_it)
