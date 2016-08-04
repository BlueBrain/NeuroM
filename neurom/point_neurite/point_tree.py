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
from neurom.core import Tree
from itertools import chain, imap, ifilter, repeat


class PointTree(Tree):
    '''
    Simple tree class. This is a recursive data structure, with each tree
    holding a value and a list of children trees. Every node is a tree.
    '''
    def __init__(self, value):
        super(PointTree, self).__init__()
        self.value = value

    def __str__(self):
        return 'Tree(value=%s) <parent: %s, nchildren: %d>' % \
            (self.value, self.parent, len(self.children))

    def isegment(self, iter_mode=Tree.ipreorder):
        '''Iterate over segments

        Segments are parent-child pairs, with the child being the
        center of the iteration

        Parameters:
            iter_mode: iteration mode. Default: ipreorder.
        '''
        return imap(lambda t: (t.parent, t),
                    ifilter(lambda t: t.parent is not None, iter_mode(self)))

    def isection(self, iter_mode=Tree.ipreorder):
        '''Iterator to sections of a tree.

        Resolves to a tuple of sub-trees forming a section.

        Parameters:
            iter_mode: iteration mode. Default: ipreorder.
        '''
        def get_section(tree):
            '''get the upstream section starting from this tree'''
            ui = tree.iupstream()
            sec = [ui.next()]
            for i in ui:
                sec.append(i)
                if i.is_forking_point() or i.is_root():
                    break
            sec.reverse()
            return tuple(sec)

        def seed_node(n):
            '''Is this node a good seed for upstream section finding?'''
            return not n.is_root() and (n.is_leaf() or n.is_forking_point())

        return imap(get_section,
                    ifilter(seed_node, iter_mode(self)))

    def itriplet(self):
        '''Iterate over triplets

        Post-order iteration yielding tuples with three consecutive sub-trees
        '''
        return chain.from_iterable(
            imap(lambda n: zip(repeat(n.parent), repeat(n), n.children),
                 ifilter(lambda n: not n.is_root() and not n.is_leaf(),
                         self.ipreorder())))

    def i_branch_end_points(self):
        '''Iterate over the furthest points in forking sections

        Returns:
            An iterator with end-points of the sections forking out\
                of fork_point
        '''

        def next_end_point(tree):
            '''Get the next node of a tree which is a section end-point
            '''
            i = tree.ipreorder()
            while True:
                n = i.next()
                if is_section_end_point(n):
                    break
            return n

        def is_section_end_point(tree):
            '''Is this tree a section end point???
            '''
            return (not tree.is_root()) and (tree.is_forking_point() or tree.is_leaf())

        return imap(next_end_point, self.children)


def as_elements(trees):
    '''Recursive Tree -> Tree.value transformation function.

    Maintains type of containing iterables
    '''
    return (type(trees)(as_elements(t) for t in trees)
            if hasattr(trees, '__iter__')
            else (trees.value if isinstance(trees, PointTree) else trees))


def val_iter(tree_iterator):
    '''Iterator adaptor to iterate over Tree.value'''
    return imap(as_elements, tree_iterator)


def imap_val(f, tree_iterator):
    '''Map function f to value of tree_iterator's target
    '''
    return imap(f, val_iter(tree_iterator))


isegment = PointTree.isegment
isection = PointTree.isection
itriplet = PointTree.itriplet
i_branch_end_points = PointTree.i_branch_end_points
