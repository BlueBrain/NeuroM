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

'''Test neurom.point_neurite.io.utils'''
import os
from itertools import izip
import numpy as np
from neurom.point_neurite.io import utils
from neurom.point_neurite import points as pts
from neurom.point_neurite import point_tree as ptree
from neurom.point_neurite.io.datawrapper import RawDataWrapper
from neurom import iter_neurites
from neurom.core.dataformat import COLS
from neurom.core import tree
from neurom.exceptions import (SomaError, IDSequenceError,
                               MultipleTrees, MissingParentError)
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')

FILES = [os.path.join(SWC_PATH, f)
         for f in ['Neuron.swc',
                   'Single_apical_no_soma.swc',
                   'Single_apical.swc',
                   'Single_basal.swc',
                   'Single_axon.swc',
                   'sequential_trunk_off_0_16pt.swc',
                   'sequential_trunk_off_1_16pt.swc',
                   'sequential_trunk_off_42_16pt.swc',
                   'Neuron_no_missing_ids_no_zero_segs.swc']]

NO_SOMA_FILE = os.path.join(SWC_PATH, 'Single_apical_no_soma.swc')

DISCONNECTED_POINTS_FILE = os.path.join(SWC_PATH, 'Neuron_disconnected_components.swc')

MISSING_PARENTS_FILE = os.path.join(SWC_PATH, 'Neuron_missing_parents.swc')

NON_CONSECUTIVE_ID_FILE = os.path.join(SWC_PATH,
                                       'non_sequential_trunk_off_1_16pt.swc')

INVALID_ID_SEQUENCE_FILE = os.path.join(SWC_PATH,
                                        'non_increasing_trunk_off_1_16pt.swc')

SOMA_IDS = [[1, 2, 3],
            [],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 9],
            [2, 10],
            [43, 51],
            [1, 2, 3]]

INIT_IDS = [[4, 215, 426, 637],
            [],
            [4],
            [4],
            [4],
            [2, 10],
            [3, 11],
            [44, 52],
            [4]]


RAW_DATA = [utils.load_data(f) for f in FILES]
NO_SOMA_RAW_DATA = utils.load_data(NO_SOMA_FILE)


class MockNeuron(object):
    def __init__(self, trees):
        self.neurites = trees


def test_get_soma_ids():
    for i, d in enumerate(RAW_DATA):
        nt.ok_(utils.get_soma_ids(d) == SOMA_IDS[i])


def test_get_initial_neurite_segment_ids():
    for i, d in enumerate(RAW_DATA):
        nt.ok_(utils.get_initial_neurite_segment_ids(d) == INIT_IDS[i])


def _check_trees(trees):
    for t in trees:
        nt.ok_(len(list(tree.ileaf(t))) == 11)
        nt.ok_(len(list(tree.iforking_point(t))) == 10)
        nt.ok_(len(list(tree.ipreorder(t))) == 211)
        nt.ok_(len(list(tree.ipostorder(t))) == 211)
        nt.ok_(len(list(ptree.isegment(t))) == 210)
        leaves = [l for l in tree.ileaf(t)]
        # path length from each leaf to root node.
        branch_order = [21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 111]
        for i, l in enumerate(leaves):
            nt.ok_(len(list(tree.iupstream(l))) == branch_order[i])


def test_make_point_tree():
    rd = RAW_DATA[0]
    seg_ids = utils.get_initial_neurite_segment_ids(rd)
    trees = [utils.make_point_tree(rd, seg_id) for seg_id in seg_ids]
    nt.ok_(len(trees) == len(INIT_IDS[0]))
    _check_trees(trees)


def test_make_tree_postaction():
    def post_action(t):
        t.foo = 'bar'

    rd = RAW_DATA[0]
    seg_ids = utils.get_initial_neurite_segment_ids(rd)
    trees = [utils.make_point_tree(rd, root_id=seg_id, post_action=post_action)
             for seg_id in seg_ids]
    for t in trees:
        nt.ok_(hasattr(t, 'foo') and t.foo == 'bar')


def test_make_neuron():
    rd = RAW_DATA[0]
    nrn = utils.make_neuron(rd)
    nt.ok_(np.all([s[COLS.ID] for s in nrn.soma.iter()] == SOMA_IDS[0]))
    _check_trees(nrn.neurites)


@nt.raises(SomaError)
def test_make_neuron_no_soma_raises_SomaError():
    utils.make_neuron(NO_SOMA_RAW_DATA)


def test_make_neuron_post_tree_action():
    def post_action(t):
        t.bar = 'foo'

    rd = RAW_DATA[0]
    nrn = utils.make_neuron(rd, post_action)
    for t in nrn.neurites:
        nt.ok_(hasattr(t, 'bar') and t.bar == 'foo')


def test_load_neuron():
    nrn = utils.load_neuron(FILES[0])
    nt.ok_(nrn.name == FILES[0].strip('.swc').split('/')[-1])


def test_load_neuron_deep_neuron():
    '''make sure that neurons with deep (ie: larger than the python
       recursion limit can be loaded)
    '''
    deep_neuron = os.path.join(DATA_PATH, 'h5/v1/deep_neuron.h5')
    utils.load_neuron(deep_neuron)


def test_load_neurolucida_ascii():
    f_ = os.path.join(DATA_PATH, 'neurolucida', 'sample.asc')
    ascii = utils.load_data(f_)
    nt.ok_(isinstance(ascii, RawDataWrapper))
    nt.eq_(len(ascii.data_block), 18)


def test_load_trees_good_neuron():
    '''Check trees in good neuron are the same as trees from loaded neuron'''
    filepath = os.path.join(SWC_PATH, 'Neuron.swc')
    nrn = utils.load_neuron(filepath)
    trees = utils.load_trees(filepath)
    nt.eq_(len(nrn.neurites), 4)
    nt.eq_(len(nrn.neurites), len(trees))

    nrn2 = MockNeuron(trees)

    @pts.point_function(as_tree=False)
    def elem(point):
        return point

    # Check data are the same in tree collection and neuron's neurites
    for a, b in izip(iter_neurites(nrn, elem), iter_neurites(nrn2, elem)):
        nt.ok_(np.all(a == b))


def test_load_trees_no_soma():

    trees = utils.load_trees(NO_SOMA_FILE)
    nt.eq_(len(trees), 1)


def test_load_trees_postaction():

    def post_action(t):
        t.foo = 'bar'

    filepath = os.path.join(SWC_PATH, 'Neuron.swc')
    trees = utils.load_trees(filepath, tree_action=post_action)
    nt.eq_(len(trees), 4)  # sanity check

    for t in trees:
        nt.ok_(hasattr(t, 'foo') and t.foo == 'bar')


def test_load_neuron_disconnected_components():

    filepath = DISCONNECTED_POINTS_FILE
    trees = utils.load_trees(filepath)
    nt.eq_(len(trees), 8)

    # tree ID - number of points map
    ref_ids_pts = {4: 1, 215: 1, 426: 1, 637: 1, 6: 209, 217: 209, 428: 209, 639: 209}

    ids_pts =  {}
    for t in trees:
        ids_pts[t.value[COLS.ID]] = pts.count(t)

    nt.eq_(ref_ids_pts, ids_pts)


@nt.raises(MultipleTrees)
def test_load_neuron_disconnected_points_raises():
    utils.load_neuron(DISCONNECTED_POINTS_FILE)


@nt.raises(MissingParentError)
def test_load_neuron_missing_parents_raises():
    utils.load_neuron(MISSING_PARENTS_FILE)


@nt.raises(SomaError)
def test_load_neuron_no_soma_raises_SomaError():
    utils.load_neuron(NO_SOMA_FILE)


@nt.raises(IDSequenceError)
def test_load_neuron_invalid_id_sequence_raises():
    utils.load_neuron(INVALID_ID_SEQUENCE_FILE);


@nt.raises(IDSequenceError)
def test_load_trees_invalid_id_sequence_raises():
    utils.load_trees(INVALID_ID_SEQUENCE_FILE);


def test_load_neuron_no_consecutive_ids_loads():
    utils.load_neuron(NON_CONSECUTIVE_ID_FILE);
