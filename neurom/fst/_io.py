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

'''Fast neuron IO module'''

import os
from collections import namedtuple, defaultdict
from functools import partial, update_wrapper
from neurom.io.hdf5 import H5
from neurom.io.swc import SWC
from neurom.io.neurolucida import NeurolucidaASC
from neurom.core.types import NeuriteType
from neurom.core.tree import Tree
from neurom.core.dataformat import POINT_TYPE
from neurom.core.dataformat import COLS
from neurom.core.neuron import make_soma
from neurom.core.population import Population
from neurom.io import utils as _iout


Neuron = namedtuple('Neuron', 'soma, neurites, data_block')


class SecDataWrapper(object):
    '''Class holding a raw data block and section information'''

    def __init__(self, data_block, fmt, sections=None):
        '''Section Data Wrapper'''
        self.data_block = data_block
        self.fmt = fmt
        self.sections = sections if sections is not None else extract_sections(data_block)

    def neurite_trunks(self):
        '''Get the section IDs of the intitial neurite sections'''
        sec = self.sections
        return [i for i, ss in enumerate(sec)
                if ss.pid > -1 and (sec[ss.pid].ntype == POINT_TYPE.SOMA and
                                    ss.ntype != POINT_TYPE.SOMA)]

    def soma_points(self):
        '''Get the soma points'''
        db = self.data_block
        return db[db[:, COLS.TYPE] == POINT_TYPE.SOMA]


_TREE_TYPES = tuple(NeuriteType)


def make_trees(rdw, post_action=None):
    '''Build section trees from a raw data wrapper'''
    trunks = rdw.neurite_trunks()
    start_node = min(trunks)

    # One pass over sections to build nodes
    nodes = [Tree(rdw.data_block[sec.ids]) for sec in rdw.sections[start_node:]]

    # One pass over nodes to set the neurite type
    # and connect children to parents
    for i in xrange(len(nodes)):
        nodes[i].type = _TREE_TYPES[rdw.sections[i + start_node].ntype]
        parent_id = rdw.sections[i + start_node].pid - start_node
        if parent_id >= 0:
            nodes[parent_id].add_child(nodes[i])

    head_nodes = [nodes[i - start_node] for i in trunks]

    if post_action is not None:
        for n in head_nodes:
            post_action(n)

    return head_nodes


def load_neuron(filename):
    '''Build section trees from an h5 or swc file'''
    _READERS = {
        'swc': lambda f: SWC.read(f, wrapper=SecDataWrapper),
        'h5': lambda f: H5.read(f, remove_duplicates=False, wrapper=SecDataWrapper),
        'asc': lambda f: NeurolucidaASC.read(f, remove_duplicates=True, wrapper=SecDataWrapper)
    }
    _NEURITE_ACTION = {
        'swc': remove_soma_initial_point,
        'h5': None,
        'asc': None
    }
    ext = os.path.splitext(filename)[1][1:]
    rdw = _READERS[ext.lower()](filename)
    trees = make_trees(rdw, _NEURITE_ACTION[ext.lower()])
    soma = make_soma(rdw.soma_points())
    return Neuron(soma, trees, rdw)


load_neurons = partial(_iout.load_neurons, neuron_loader=load_neuron)
update_wrapper(load_neurons, _iout.load_neurons)


load_population = partial(_iout.load_population, neuron_loader=load_neurons,
                          population_class=Population)
update_wrapper(load_population, _iout.load_population)


def extract_sections(data_block):
    '''Make a list of sections from an SWC-style data wrapper block'''

    class Section(object):
        '''sections ((ids), type, parent_id)'''
        def __init__(self, ids=None, ntype=0, pid=-1):
            self.ids = [] if ids is None else ids
            self.ntype = ntype
            self.pid = pid

    # get SWC ID to array position map
    id_map = {-1: -1}
    for i, r in enumerate(data_block):
        id_map[int(r[COLS.ID])] = i

    # number of children per point
    n_children = defaultdict(int)
    for row in data_block:
        n_children[int(row[COLS.P])] += 1

    # end points have either no children or more than one
    sec_end_pts = set(i for i, row in enumerate(data_block)
                      if n_children[row[COLS.ID]] != 1)

    _sections = [Section()]
    curr_section = _sections[-1]
    parent_section = {-1: -1}

    for row in data_block:
        row_id = id_map[int(row[COLS.ID])]
        if len(curr_section.ids) == 0:
            curr_section.ids.append(id_map[int(row[COLS.P])])
            curr_section.ntype = int(row[COLS.TYPE])
        curr_section.ids.append(row_id)
        if row_id in sec_end_pts:
            parent_section[curr_section.ids[-1]] = len(_sections) - 1
            _sections.append(Section())
            curr_section = _sections[-1]

    # get the section parent ID from the id of the first point.
    for sec in _sections:
        if sec.ids:
            sec.pid = parent_section[sec.ids[0]]

    return [s for s in _sections if s.ids]


def remove_soma_initial_point(tree):
    '''Remove tree's initial point if soma'''
    if tree.value[0][COLS.TYPE] == POINT_TYPE.SOMA:
        tree.value = tree.value[1:]
