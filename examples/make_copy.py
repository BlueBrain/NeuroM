#!/usr/bin/env python
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

''' Copy function for trees
'''
from copy import copy
from neurom.core.tree import Tree


def make_copy(tree):
    '''
    Copy a tree structure

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

if __name__ == "__main__":

    import nose.tools as nt
    from neurom.ezy import load_neuron
    from neurom.analysis.morphtree import compare_trees

    def test_make_copy():

        n = load_neuron('test_data/valid_set/Neuron.swc').neurites[0]

        m = make_copy(n)

        # first check if they are identical

        nt.assert_true(compare_trees(n, m))

        # check if they refer to the same value

        n.children[0].value[1] = - 5000.

        nt.assert_false(compare_trees(n, m))

        n.value[0] = -10000.

        nt.assert_false(n.value[0] == m.value[0])

    print "Running Test"
    test_make_copy()
    print "Finished"
