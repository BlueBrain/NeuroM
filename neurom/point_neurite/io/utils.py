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

'''Utility functions and classes for building point-tree neurons'''
import itertools
from functools import partial
from neurom.io import load_neurons as _load_neurons
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE
from neurom.core.dataformat import ROOT_ID
from neurom.point_neurite.point_tree import PointTree
from neurom.point_neurite.treefunc import set_tree_type
from neurom.point_neurite.core import PointNeuron
from neurom.point_neurite.io.datawrapper import DataWrapper
from neurom.core import make_soma
from neurom.exceptions import IDSequenceError, MultipleTrees, MissingParentError
from neurom.check import structural_checks as check
from neurom.utils import memoize
import os


@memoize
def get_soma_ids(rdw):
    '''Returns a list of IDs of points that are somas'''
    return rdw.soma_points()[:, COLS.ID].tolist()


@memoize
def get_initial_neurite_segment_ids(rdw):
    '''Returns a list of IDs of initial neurite tree segments

    These are defined as non-soma points whose parent is a soma point.
    '''
    l = list(itertools.chain.from_iterable([rdw.get_children(s) for s in get_soma_ids(rdw)]))
    return [i for i in l if rdw.get_row(i)[COLS.TYPE] != POINT_TYPE.SOMA]


def make_point_tree(rdw, root_id=ROOT_ID, post_action=None):
    '''Return a tree obtained from a raw data block

    The tree contains rows of raw data.
    Args:
        rdw: a DataWrapper object.
        root_id: ID of the root of the tree to be built.
        post_action: optional function to run on the built tree.
    '''
    head_node = PointTree(rdw.get_row(root_id))
    children = [head_node, ]
    while children:
        cur_node = children.pop()
        for c in rdw.get_children(cur_node.value[COLS.ID]):
            row = rdw.get_row(c)
            child = PointTree(row)
            cur_node.add_child(child)
            children.append(child)

    if post_action is not None:
        post_action(head_node)

    return head_node


def make_neuron(raw_data, tree_action=None):
    '''Build a neuron from a raw data block

    The tree contains rows of raw data.
    Parameters:
        raw_data: a DataWrapper object.
        tree_action: optional function to run on the built trees.
    Raises:
        SomaError if no soma points in raw_data or points incompatible with soma.
        IDSequenceError if filename contains invalid ID sequence
    '''
    _soma = make_soma(raw_data.soma_points())
    _trees = [make_point_tree(raw_data, iseg, tree_action)
              for iseg in get_initial_neurite_segment_ids(raw_data)]

    nrn = PointNeuron(_soma, _trees)
    nrn.data_block = raw_data
    return nrn


def load_neuron(filename, tree_action=set_tree_type):
    """
    Loads a neuron keeping a record of the filename.
    Args:
        filename: the path of the file storing morphology data
        tree_action: optional function to run on each of the neuron's
        neurite trees.
    Raises:
        SomaError if no soma points in data.
        IDSequenceError if filename contains invalid ID sequence
    """

    data = load_data(filename)
    if not check.has_increasing_ids(data):
        raise IDSequenceError('Invald ID sequence found in raw data')
    if not check.is_single_tree(data):
        raise MultipleTrees('Multiple trees detected')
    if not check.no_missing_parents(data):
        raise MissingParentError('Missing parents detected')

    nrn = make_neuron(data, tree_action)
    nrn.name = os.path.splitext(os.path.basename(filename))[0]

    return nrn


def load_trees(filename, tree_action=None):
    """Load all trees in an input file

    Loads all trees, regardless of whether they are connected
    Args:
        filename: the path of the file storing morphology data
        tree_action: optional function to run on each of the neuron's
        neurite trees.
    Raises:
        IDSequenceError if filename contains non-incremental ID sequence
    """
    data = load_data(filename)

    if not check.has_increasing_ids(data):
        raise IDSequenceError('Invald ID sequence found in raw data')

    _ids = get_initial_neurite_segment_ids(data)
    _ids.extend(data.get_ids(lambda r: r[COLS.P] == -1 and r[COLS.TYPE] != POINT_TYPE.SOMA))

    return [make_point_tree(data, i, tree_action) for i in _ids]


def load_data(filename):
    '''Unpack filename and return a DataWrapper object containing the data

    Determines format from extension. Currently supported:

        * SWC (case-insensitive extension ".swc")
        * H5 v1 and v2 (case-insensitive extension ".h5"). Attempts to
          determine the version from the contents of the file
        * Neurolucida ASCII (case-insensitive extension ".asc")
    '''

    def read_h5(filename):
        '''Lazy loading of HDF5 reader'''
        from neurom.io import hdf5
        return hdf5.read(filename,
                         remove_duplicates=True,
                         data_wrapper=DataWrapper)

    def read_swc(filename):
        '''Lazy loading of SWC reader'''
        from neurom.io import swc
        from .swc import SWCDataWrapper
        return swc.read(filename, data_wrapper=SWCDataWrapper)

    def read_neurolucida(filename):
        '''Lazy loading of Neurolucida ASCII reader'''
        from neurom.io import neurolucida
        return neurolucida.read(filename,
                                data_wrapper=DataWrapper)

    _READERS = {
        'swc': read_swc,
        'h5': read_h5,
        'asc': read_neurolucida,
    }
    extension = os.path.splitext(filename)[1][1:]
    return _READERS[extension.lower()](filename)


load_neurons = partial(_load_neurons, neuron_loader=load_neuron)
