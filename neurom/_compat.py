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

'''Old-style new-style neurite compatibility hacks'''

from itertools import imap, izip
from neurom import segments as seg
from neurom import iter_neurites
from neurom.core.tree import ipreorder, Tree
from neurom.analysis.morphtree import get_bounding_box
from neurom.analysis.morphtree import find_tree_type
from neurom import fst


def is_new_style(obj):
    '''Determine whether a neuron or tree is new or old style'''
    if isinstance(obj, fst.Neuron):
        return True
    elif isinstance(obj, Tree):
        return len(obj.value.shape) == 2
    else:
        return False


def bounding_box(tree):
    '''Get a tree's X,Y,Z bounding box'''
    if is_new_style(tree):
        return fst._mm.bounding_box(tree)  # pylint: disable=protected-access
    else:
        return get_bounding_box(tree)


def neurite_type(tree):
    '''Get the neurite type of a neurite tree'''
    if is_new_style(tree):
        return tree.type
    else:
        return find_tree_type(tree)


def map_segments(tree, fun):
    '''map a function to the segments in a tree'''
    def _segfun(sec):
        '''map a segment function to the segments in section sec'''
        return imap(fun, izip(sec.value[:-1], sec.value[1:]))

    if is_new_style(tree):
        return [s for ss in ipreorder(tree) for s in _segfun(ss)]
    else:
        fun = seg.segment_function(as_tree=False)(fun)
        return list(iter_neurites(tree, fun))
