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
from neurom.core.neuron import Neuron, make_soma
from neurom.core.population import Population
from neurom.exceptions import IDSequenceError, MultipleTrees, MissingParentError
from . import load_data
from . import check
from neurom.utils import memoize
import os


@memoize
def get_soma_ids(rdw):
    '''Returns a list of IDs of points that are somas'''
    return rdw.get_soma_rows()[:, COLS.ID].tolist()


@memoize
def get_initial_neurite_segment_ids(rdw):
    '''Returns a list of IDs of initial neurite tree segments

    These are defined as non-soma points whose parent is a soma point.
    '''
    l = list(itertools.chain.from_iterable([rdw.get_children(s) for s in get_soma_ids(rdw)]))
    return [i for i in l if rdw.get_row(i)[COLS.TYPE] != POINT_TYPE.SOMA]


def make_tree(rdw, root_id=ROOT_ID, post_action=None):
    '''Return a tree obtained from a raw data block

    The tree contains rows of raw data.
    Args:
        rdw: a RawDataWrapper object.
        root_id: ID of the root of the tree to be built.
        post_action: optional function to run on the built tree.
    '''
    head_node = Tree(rdw.get_row(root_id))
    children = [head_node, ]
    while children:
        cur_node = children.pop()
        for c in rdw.get_children(cur_node.value[COLS.ID]):
            row = rdw.get_row(c)
            child = Tree(row)
            cur_node.add_child(child)
            children.append(child)

    if post_action is not None:
        post_action(head_node)

    return head_node


def make_neuron(raw_data, tree_action=None):
    '''Build a neuron from a raw data block

    The tree contains rows of raw data.
    Parameters:
        raw_data: a RawDataWrapper object.
        tree_action: optional function to run on the built trees.
    Raises:
        SomaError if no soma points in raw_data or points incompatible with soma.
        IDSequenceError if filename contains invalid ID sequence
    '''
    _soma = make_soma(raw_data.get_soma_rows())
    _trees = [make_tree(raw_data, iseg, tree_action)
              for iseg in get_initial_neurite_segment_ids(raw_data)]

    return Neuron(_soma, _trees)


def load_neuron(filename, tree_action=None):
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
    if not check.has_increasing_ids(data)[0]:
        raise IDSequenceError('Invald ID sequence found in raw data')
    if not check.is_single_tree(data)[0]:
        raise MultipleTrees('Multiple trees detected')
    if not check.no_missing_parents(data)[0]:
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

    if not check.has_increasing_ids(data)[0]:
        raise IDSequenceError('Invald ID sequence found in raw data')

    _ids = get_initial_neurite_segment_ids(data)
    _ids.extend(data.get_ids(lambda r: r[COLS.P] == -1 and r[COLS.TYPE] != POINT_TYPE.SOMA))

    return [make_tree(data, i, tree_action) for i in _ids]


def get_morph_files(directory):
    '''Get a list of all morphology files in a directory

    Returns:
        list with all files with extensions '.swc' , 'h5' or '.asc' (case insensitive)
    '''
    lsdir = [os.path.join(directory, m) for m in os.listdir(directory)]
    return [m for m in lsdir
            if os.path.isfile(m) and
            os.path.splitext(m)[1].lower() in ('.swc', '.h5', '.asc')]


def load_neurons(neurons, neuron_loader=load_neuron):
    '''Create a list of Neuron objects from each morphology file in directory\
        or from a list or tuple of file names

    Parameters:
        neurons: directory path or list of neuron file paths

    Returns:
        list of Neuron objects
    '''
    if isinstance(neurons, list) or isinstance(neurons, tuple):
        return [neuron_loader(f) for f in neurons]
    elif isinstance(neurons, str):
        return [neuron_loader(f) for f in get_morph_files(neurons)]


def load_population(neurons, name=None, neuron_loader=load_neurons,
                    population_class=Population):
    '''Create a population object from all morphologies in a directory\
        of from morphologies in a list of file names

    Parameters:
        neurons: directory path or list of neuron file paths
        population_class: class representing populations
        name (str): optional name of population. By default 'Population' or\
            filepath basename depending on whether neurons is list or\
            directory path respectively.

    Returns:
        neuron population object

    '''
    pop = population_class(neuron_loader(neurons))
    if isinstance(neurons, list) or isinstance(neurons, tuple):
        name = name if name is not None else 'Population'
    elif isinstance(neurons, str):
        name = name if name is not None else os.path.basename(neurons)

    pop.name = name
    return pop
