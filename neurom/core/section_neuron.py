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

'''Section tree module'''

import math
from collections import defaultdict
from collections import namedtuple
import numpy as np
from neurom.io.hdf5 import H5
from neurom.core.tree import Tree, ipreorder
from neurom.core.dataformat import POINT_TYPE
from neurom.core.dataformat import COLS
from neurom.core.tree import i_chain2, iupstream
from neurom.core.neuron import make_soma
from neurom.analysis import morphmath as mm


class SEC(object):
    '''Enum class with section content indices'''
    (START, END, TYPE, ID, PID) = xrange(5)


Neuron = namedtuple('Neuron', 'soma, neurites, data_block')


class SecDataWrapper(object):
    '''Class holding a raw data block and section information'''

    def __init__(self, data_block, fmt, sections):
        '''Section Data Wrapper'''
        self.data_block = data_block
        self.fmt = fmt
        self.sections = sections
        self.adj_list = defaultdict(list)

        for sec in self.sections:
            self.adj_list[sec[SEC.PID]].append(sec[SEC.ID])

    def neurite_trunks(self):
        '''Get the section IDs of the intitial neurite sections'''
        sec = self.sections
        return [ss[SEC.ID] for ss in sec
                if sec[ss[SEC.PID]][SEC.TYPE] == POINT_TYPE.SOMA and
                ss[SEC.TYPE] != POINT_TYPE.SOMA]

    def soma_points(self):
        '''Get the soma points'''
        db = self.data_block
        return db[db[:, COLS.TYPE] == POINT_TYPE.SOMA]


def make_tree(rdw, root_node=0):
    '''Build a section tree'''
    _sec = rdw.sections
    head_node = Tree(_sec[root_node])
    children = [head_node]
    while children:
        cur_node = children.pop()
        for c in rdw.adj_list[cur_node.value[SEC.ID]]:
            child = Tree(_sec[c])
            cur_node.add_child(child)
            children.append(child)

    # set the data in each node to a slice of the raw data block
    for n in ipreorder(head_node):
        sec = n.value
        n.value = rdw.data_block[sec[SEC.START]: sec[SEC.END]]

    return head_node


def load_neuron(filename):
    '''Build section trees from an h5 file'''
    rdw = H5.read(filename, remove_duplicates=False, wrapper=SecDataWrapper)
    trees = [make_tree(rdw, trunk) for trunk in rdw.neurite_trunks()]
    soma = make_soma(rdw.soma_points())
    return Neuron(soma, trees, rdw)


def soma_surface_area(nrn):
    '''Get the surface area of a neuron's soma.

    Note:
        The surface area is calculated by assuming the soma is spherical.
    '''
    return 4 * math.pi * nrn.soma.radius ** 2


def n_segments(nrn):
    '''Number of segments in a section'''
    return sum(len(s.value) - 1 for s in i_chain2(nrn.neurites))


def n_sections(nrn):
    '''Number of sections in a neuron'''
    return sum(1 for _ in i_chain2(nrn.neurites))


def n_neurites(nrn):
    '''Number of neurites in a neuron'''
    return len(nrn.neurites)


def path_length(section):
    '''Path length from section to root'''
    return sum(mm.path_distance(s.value) for s in iupstream(section))


def get_section_lengths(nrn):
    '''section lengths'''
    return np.array([mm.path_distance(s.value) for s in i_chain2(nrn.neurites)])


def get_path_lengths(nrn):
    '''Less naive path length calculation

    Calculates and stores the section lengths in one pass,
    then queries the lengths in the path length iterations.
    This avoids repeatedly calculating the lengths of the
    same sections.
    '''
    dist = {}

    for s in i_chain2(nrn.neurites):
        dist[s] = mm.path_distance(s.value)

    def pl2(sec):
        '''Calculate the path length using cahced section lengths'''
        return sum(dist[s] for s in iupstream(sec))

    return np.array([pl2(s) for s in i_chain2(nrn.neurites)])
