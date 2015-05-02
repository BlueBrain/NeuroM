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

'''Neuron building example.

An example of how to build an object representing a neuron from an SWC file
'''
from itertools import imap
from neurom.io.readers import load_data
from neurom.io.utils import make_tree
from neurom.io.utils import make_neuron
from neurom.io.utils import get_soma_ids
from neurom.io.utils import get_initial_segment_ids
from neurom.core import tree
from neurom.core import neuron
from neurom.core.dataformat import COLS
from neurom.core.point import as_point


def point_iter(iterator):
    '''Transform tree iterator into a point iterator

    Args:
        iterator: tree iterator for a tree holding raw data rows.
    '''
    return imap(as_point, tree.val_iter(iterator))


if __name__ == '__main__':

    filename = 'test_data/swc/Neuron.swc'

    rd = load_data(filename)

    init_seg_ids = get_initial_segment_ids(rd)

    trees = [make_tree(rd, sg) for sg in init_seg_ids]

    soma_pts = [rd.get_row(si) for si in get_soma_ids(rd)]

    for tr in trees:
        for p in point_iter(tree.iter_preorder(tr)):
            print p

    print 'Initial segment IDs:', init_seg_ids

    nrn = neuron.Neuron(soma_pts, trees)

    print 'Neuron soma raw data', [r for r in nrn.soma.iter()]
    print 'Neuron soma points', [as_point(p)
                                 for p in nrn.soma.iter()]

    print 'Neuron tree init points, types'
    for tt in nrn.neurite_trees:
        print tt.value[COLS.ID], tt.value[COLS.TYPE]

    print 'Making neuron 2'
    nrn2 = make_neuron(rd)
    print 'Neuron 2 soma points', [r for r in nrn2.soma.iter()]
    print 'Neuron 2 soma points', [as_point(p)
                                   for p in nrn2.soma.iter()]
    print 'Neuron 2 tree init points, types'
    for tt in nrn2.neurite_trees:
        print tt.value[COLS.ID], tt.value[COLS.TYPE]

    print 'Print neuron leaves as points'
    for tt in nrn2.neurite_trees:
        for p in point_iter(tree.iter_leaf(tt)):
            print p
