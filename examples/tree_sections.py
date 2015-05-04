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
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''Build sections from a tree

Sections are defined as points between forking points,
between the root node and forking points, or between
forking points and end-points

'''

from itertools import imap
from itertools import ifilter
from neurom.core import tree

REF_TREE = tree.Tree(0)
REF_TREE.add_child(tree.Tree(11))
REF_TREE.add_child(tree.Tree(12))
REF_TREE.children[0].add_child(tree.Tree(111))
REF_TREE.children[0].add_child(tree.Tree(112))
REF_TREE.children[1].add_child(tree.Tree(121))
REF_TREE.children[1].add_child(tree.Tree(122))
REF_TREE.children[1].children[0].add_child(tree.Tree(1211))
REF_TREE.children[1].children[0].children[0].add_child(tree.Tree(12111))
REF_TREE.children[1].children[0].children[0].add_child(tree.Tree(12112))
REF_TREE.children[0].children[0].add_child(tree.Tree(1111))
REF_TREE.children[0].children[0].children[0].add_child(tree.Tree(11111))
REF_TREE.children[0].children[0].children[0].add_child(tree.Tree(11112))


def is_forking_point(t):
    '''Is this tree a forking point?'''
    return len(t.children) > 1


def is_leaf(t):
    '''Is this tree a leaf?'''
    return len(t.children) == 0


def is_root(t):
    '''Is this tree the root node?'''
    return t.parent is None


def get_section(t):
    '''get the upstream section starting from this tree'''
    ui = tree.iter_upstream(t)
    sec = [ui.next()]
    for i in ui:
        sec.append(i)
        if is_forking_point(i) or is_root(i):
            break
    sec.reverse()
    return tuple(sec)


def iter_section(t):
    '''Iterator to sections of a tree.

    Resolves to a tuple of sub-trees forming a section.
    '''
    def boundary_node(n):
        '''Is this a section boundary node?'''
        return not is_root(n) and (is_leaf(n) or is_forking_point(n))

    return imap(get_section,
                ifilter(boundary_node, tree.iter_preorder(t)))


if __name__ == '__main__':

    for s in iter_section(REF_TREE):
        print [tt.value for tt in s]
