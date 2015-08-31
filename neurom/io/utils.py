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

'''Utility functions and classes for higher level RawDataWrapper access'''
import itertools
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE
from neurom.core.dataformat import ROOT_ID
from neurom.core.tree import Tree
from neurom.core.neuron import Neuron
from neurom.exceptions import SomaError
from neurom.exceptions import NonConsecutiveIDsError
from neurom.io.readers import load_data
from neurom.io.check import has_sequential_ids
import os


def get_soma_ids(rdw):
    '''Returns a list of IDs of points that are somas'''
    return rdw.get_ids(lambda r: r[COLS.TYPE] == POINT_TYPE.SOMA)


def get_initial_segment_ids(rdw):
    '''Returns a list of IDs of initial tree segments

    These are defined as non-soma points whose perent is a soma point.
    '''
    l = list(itertools.chain(*[rdw.get_children(s) for s in get_soma_ids(rdw)]))
    return [i for i in l if rdw.get_row(i)[COLS.TYPE] != POINT_TYPE.SOMA]


def make_tree(rdw, root_id=ROOT_ID, post_action=None):
    '''Return a tree obtained from a raw data block

    The tree contains rows of raw data.
    Args:
        rdw: a RawDataWrapper object.
        root_id: ID of the root of the tree to be built.
        post_action: optional function to run on the built tree.
    '''
    def add_children(t):
        '''Add children to a tree'''
        for c in rdw.get_children(t.value[COLS.ID]):
            child = Tree(rdw.get_row(c))
            t.add_child(child)
            add_children(child)
        return t

    head_node = Tree(rdw.get_row(root_id))
    add_children(head_node)

    if post_action is not None:
        post_action(head_node)

    return head_node


def make_neuron(raw_data, tree_action=None):
    '''Build a neuron from a raw data block

    The tree contains rows of raw data.
    Args:
        raw_data: a RawDataWrapper object.
        tree_action: optional function to run on the built trees.
    Raises: SomaError if no soma points in raw_data.
    '''
    _trees = [make_tree(raw_data, iseg, tree_action)
              for iseg in get_initial_segment_ids(raw_data)]
    _soma_pts = [raw_data.get_row(s_id) for s_id in get_soma_ids(raw_data)]
    if not _soma_pts:
        raise SomaError('No soma points found in data')
    return Neuron(_soma_pts, _trees)


def load_neuron(filename, tree_action=None):
    """
    Loads a neuron keeping a record of the filename.
    Args:
        filename: the path of the file storing morphology data
        tree_action: optional function to run on each of the neuron's
        neurite trees.
    Raises: SomaError if no soma points in data.
    Raises: NonConsecutiveIDsError if filename contains non-consecutive
    point IDs
    """

    data = load_data(filename)
    if not has_sequential_ids(data)[0]:
        raise NonConsecutiveIDsError('Non consecutive IDs found in raw data')

    nrn = make_neuron(data, tree_action)
    nrn.id = os.path.splitext(filename)[0]

    return nrn


def get_morph_files(directory):
    '''Get a list of all morphology files in a directory

    Returns:
        list with all files with extensions '.swc' or '.h5' (case insensitive)
    '''
    lsdir = [os.path.join(directory, m) for m in os.listdir(directory)]
    return [m for m in lsdir
            if os.path.isfile(m) and
            os.path.splitext(m)[1].lower() in ('.swc', '.h5')]
