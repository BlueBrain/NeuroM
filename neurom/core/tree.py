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
from itertools import chain, imap, ifilter, repeat


class Tree(object):
    '''
    Simple tree class. This is a recursive data structure, with each tree
    holding a value and a list of children trees. Every node is a tree.
    '''
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = list()

    def add_child(self, tree):
        '''Add a child to the list of this tree's children

        This tree becomes the added tree's parent
        '''
        self.children.append(tree)
        tree.parent = self
        return tree


def is_forking_point(tree):
    '''Is tree a forking point?'''
    return len(tree.children) > 1


def is_bifurcation_point(tree):
    '''Is tree a bifurcation point?'''
    return len(tree.children) == 2


def is_leaf(tree):
    '''Is tree a leaf?'''
    return len(tree.children) == 0


def is_root(tree):
    '''Is tree the root node?'''
    return tree.parent is None


def ipreorder(tree):
    '''Depth-first pre-order iteration of tree nodes'''
    yield tree
    for v in chain(*imap(ipreorder, tree.children)):
        yield v


def ipostorder(tree):
    '''Depth-first post-order iteration of tree nodes'''
    for v in chain(*imap(ipostorder, tree.children)):
        yield v
    yield tree


def iupstream(tree):
    '''Iterate from a tree node to the root nodes'''
    t = tree
    while t is not None:
        yield t
        t = t.parent


def ileaf(tree):
    '''Iterator to all leaves of a tree'''
    return ifilter(is_leaf, ipreorder(tree))


def iforking_point(tree, iter_mode=ipreorder):
    '''Iterator to forking points. Returns a tree object.

    Parameters:
        tree: the tree over which to iterate
        iter_mode: iteration mode. Default: ipreorder.
    '''
    return ifilter(is_forking_point, iter_mode(tree))


def ibifurcation_point(tree, iter_mode=ipreorder):
    '''Iterator to bifurcation points. Returns a tree object.

    Parameters:
        tree: the tree over which to iterate
        iter_mode: iteration mode. Default: ipreorder.
    '''
    return ifilter(is_bifurcation_point, iter_mode(tree))


def isegment(tree, iter_mode=ipreorder):
    '''Iterate over segments

    Segments are parent-child pairs, with the child being the
    center of the iteration

    Parameters:
        tree: the tree over which to iterate
        iter_mode: iteration mode. Default: ipreorder.
    '''
    return imap(lambda t: (t.parent, t),
                ifilter(lambda t: t.parent is not None, iter_mode(tree)))


def isection(tree, iter_mode=ipreorder):
    '''Iterator to sections of a tree.

    Resolves to a tuple of sub-trees forming a section.

    Parameters:
        tree: the tree over which to iterate
        iter_mode: iteration mode. Default: ipreorder.
    '''
    def get_section(tree):
        '''get the upstream section starting from this tree'''
        ui = iupstream(tree)
        sec = [ui.next()]
        for i in ui:
            sec.append(i)
            if is_forking_point(i) or is_root(i):
                break
        sec.reverse()
        return tuple(sec)

    def seed_node(n):
        '''Is this node a good seed for upstream section finding?'''
        return not is_root(n) and (is_leaf(n) or is_forking_point(n))

    return imap(get_section,
                ifilter(seed_node, iter_mode(tree)))


def itriplet(tree):
    '''Iterate over triplets

    Post-order iteration yielding tuples with three consecutive sub-trees
    '''
    return chain(
        *imap(lambda n: zip(repeat(n.parent), repeat(n), n.children),
              ifilter(lambda n: not is_root(n) and not is_leaf(n),
                      ipreorder(tree))))


def i_branch_end_points(fork_point):
    '''Iterate over the furthest points in forking sections

    Parameters:
        fork_point: A tree object which is a forking point

    Returns:
        An iterator with end-points of the sections forking out\
            of fork_point
    '''

    def next_end_point(tree):
        '''Get the next node of a tree which is a section end-point
        '''
        i = ipreorder(tree)
        while True:
            n = i.next()
            if is_section_end_point(n):
                break
        return n

    def is_section_end_point(tree):
        '''Is this tree a section end point???
        '''
        return (not is_root(tree)) and (is_forking_point(tree) or is_leaf(tree))

    return imap(next_end_point, fork_point.children)


def val_iter(tree_iterator):
    '''Iterator adaptor to iterate over Tree.value'''
    def _deep_map(f, data):
        '''Recursive map function. Maintains type of iterables'''
        return (type(data)(_deep_map(f, x) for x in data)
                if hasattr(data, '__iter__')
                else f(data))
    return imap(lambda t: _deep_map(lambda n: n.value, t), tree_iterator)


def imap_val(f, tree_iterator):
    '''Map function f to value of tree_iterator's target
    '''
    return imap(f, val_iter(tree_iterator))


def i_chain(trees, iterator_type, mapping=None, tree_filter=None):
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

    chain_it = chain(*imap(iterator_type, nrt))
    return chain_it if mapping is None else imap_val(mapping, chain_it)
