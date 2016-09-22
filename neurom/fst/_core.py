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

from copy import deepcopy
import numpy as np
from neurom.core import NeuriteType
from neurom.core import make_soma, SomaError
from neurom.core import Section, Neurite, Neuron
from neurom.core.dataformat import POINT_TYPE, COLS, ROOT_ID


class FstNeuron(Neuron):
    '''Class representing a neuron'''
    def __init__(self, data_wrapper, name='Neuron'):
        self._data = data_wrapper
        neurites, sections = make_neurites(self._data)
        soma = make_soma(self._data.soma_points(), _SOMA_ACTION[self._data.fmt])
        super(FstNeuron, self).__init__(soma, neurites, sections)
        self.name = name
        self._points = None

    @property
    def points(self):
        '''Return unordered array with all the points in this neuron'''
        if self._points is None:
            _points = self.soma.points.tolist()
            for n in self.neurites:
                _points.extend(n.points.tolist())
            self._points = np.array(_points)

        return self._points

    def transform(self, trans):
        '''Return a copy of this neuron with a 3D transformation applied'''
        _data = deepcopy(self._data)
        _data.data_block[:, 0:3] = trans(_data.data_block[:, 0:3])
        return FstNeuron(_data, self.name)

    def __deepcopy__(self, memo):
        '''Deep-copy neuron object

        Note:
            Efficient copying is performed by deep-copying the internal
            data block and building a neuron from it
        '''
        return FstNeuron(deepcopy(self._data, memo), self.name)


def make_neurites(rdw):
    '''Build neurite trees from a raw data wrapper'''
    post_action = _NEURITE_ACTION[rdw.fmt]
    trunks = rdw.neurite_root_section_ids()
    if len(trunks) == 0:
        return [], []

    # One pass over sections to build nodes
    nodes = tuple(Section(section_id=i,
                          points=rdw.data_block[sec.ids],
                          section_type=_TREE_TYPES[sec.ntype])
                  for i, sec in enumerate(rdw.sections))

    # One pass over nodes to connect children to parents
    for i, node in enumerate(nodes):
        parent_id = rdw.sections[i].pid
        parent_type = nodes[parent_id].type
        # only connect neurites
        if parent_id != ROOT_ID and parent_type != NeuriteType.soma:
            nodes[parent_id].add_child(node)

    neurites = tuple(Neurite(nodes[i]) for i in trunks)

    if post_action is not None:
        for n in neurites:
            post_action(n.root_node)

    return neurites, nodes


def _remove_soma_initial_point(tree):
    '''Remove tree's initial point if soma'''
    if tree.points[0][COLS.TYPE] == POINT_TYPE.SOMA:
        tree.points = tree.points[1:]


def _check_soma_topology_swc(points):
    '''check if points form valid soma

    Currently checks if there are bifurcations within a soma
    with more than three points.
    '''
    if len(points) == 3:
        return

    parents = tuple(p[COLS.P] for p in points if p[COLS.P] != ROOT_ID)
    if len(parents) > len(set(parents)):
        raise SomaError("Bifurcating soma")


_TREE_TYPES = tuple(NeuriteType)

_NEURITE_ACTION = {
    'SWC': _remove_soma_initial_point,
    'H5V1': None,
    'H5V2': None,
    'NL-ASCII': None
}

_SOMA_ACTION = {
    'SWC': _check_soma_topology_swc,
    'H5V1': None,
    'H5V2': None,
    'NL-ASCII': None
}
