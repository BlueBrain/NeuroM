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
from collections import deque
from copy import copy


class Tree(object):
    '''
    Simple tree class. This is a recursive data structure, with each tree
    holding a value and a list of children trees. Every node is a tree.
    '''
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = list()

    def __str__(self):
        return 'Tree(value=%s) <parent: %s, nchildren: %d>' % \
            (self.value, self.parent, len(self.children))

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
    children = deque((tree, ))
    while children:
        cur_node = children.pop()
        children.extend(reversed(cur_node.children))
        yield cur_node


def ipostorder(tree):
    '''Depth-first post-order iteration of tree nodes'''
    children = [tree, ]
    seen = set()
    while children:
        cur_node = children[-1]
        if cur_node not in seen:
            seen.add(cur_node)
            children.extend(reversed(cur_node.children))
        else:
            children.pop()
            yield cur_node


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
    return chain.from_iterable(
        imap(lambda n: zip(repeat(n.parent), repeat(n), n.children),
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


def as_elements(trees):
    '''Recursive Tree -> Tree.value transformation function.

    Maintains type of containing iterables
    '''
    return (type(trees)(as_elements(t) for t in trees)
            if hasattr(trees, '__iter__')
            else (trees.value if isinstance(trees, Tree) else trees))


def val_iter(tree_iterator):
    '''Iterator adaptor to iterate over Tree.value'''
    return imap(as_elements, tree_iterator)


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

    chain_it = chain.from_iterable(imap(iterator_type, nrt))
    return chain_it if mapping is None else imap_val(mapping, chain_it)


def i_chain2(trees, iterator_type=ipreorder, mapping=None, tree_filter=None):
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


def make_copy(tree):
    '''
    Copies a tree structure. A new tree is generated with the copied values
    and node structure as the input one and is returned.

    Input : tree object

    Returns : copied tree object
    '''
    copy_head = Tree(copy(tree.value))

    orig_children = [tree, ]
    copy_children = [copy_head, ]

    while orig_children:

        orig_current_node = orig_children.pop()
        copy_current_node = copy_children.pop()

        for c in orig_current_node.children:

            copy_child = Tree(copy(c.value))

            copy_current_node.add_child(copy_child)

            orig_children.append(c)
            copy_children.append(copy_child)

    return copy_head
