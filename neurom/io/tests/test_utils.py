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

'''Test neurom.io.utils'''
import os
import numpy as np
from neurom.core import Neuron, SomaError
from neurom.fst import _neuritefunc as _nf
from neurom import get
from neurom.io import utils
from neurom.exceptions import RawDataError, SomaError, NeuroMError
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')
VALID_DATA_PATH = os.path.join(_path, DATA_PATH, 'valid_set')

NRN_NAMES = ('Neuron', 'Neuron_h5v1', 'Neuron_h5v2')

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

FILENAMES = [os.path.join(VALID_DATA_PATH, f)
             for f in ['Neuron.swc', 'Neuron_h5v1.h5', 'Neuron_h5v2.h5']]

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


def _get_name(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def _mock_load_neuron(filename):
    class MockNeuron(object):
        def __init__(self, name):
            self.soma = 42
            self.neurites = list()
            self.name = name

    return MockNeuron(_get_name(filename))


def _check_neurites_have_no_parent(nrn):

    for n in nrn.neurites:
        nt.assert_true(n.root_node.parent is None)


def test_load_neurons():
    nrns = utils.load_neurons(FILES, neuron_loader=_mock_load_neuron)
    for i, nrn in enumerate(nrns):
        nt.assert_equal(nrn.name, _get_name(FILES[i]))


def test_get_morph_files():
    ref = set(['Neuron_h5v2.h5', 'Neuron_2_branch_h5v2.h5',
               'Neuron.swc', 'Neuron_h5v1.h5', 'Neuron_2_branch_h5v1.h5'])

    FILE_PATH = os.path.abspath(os.path.join(DATA_PATH, 'valid_set'))
    files = set(os.path.basename(f) for f in utils.get_morph_files(FILE_PATH))

    nt.assert_equal(ref, files)



def test_load_neuron():

    nrn = utils.load_neuron(FILENAMES[0])
    nt.assert_true(isinstance(NRN, Neuron))
    nt.assert_equal(NRN.name, 'Neuron')
    _check_neurites_have_no_parent(nrn)


def test_neuron_name():

    for fn, nn in zip(FILENAMES, NRN_NAMES):
        nrn = utils.load_neuron(fn)
        nt.eq_(nrn.name, nn)


def test_load_contour_soma_neuron():
    nrn = utils.load_neuron(os.path.join(SWC_PATH, 'soma', 'contour_soma_neuron.swc'))
    nt.eq_(len(nrn.neurites), 3)
    nt.eq_(len(nrn.soma.points), 8)
    nt.eq_(nrn.soma.radius, 2.125)
    _check_neurites_have_no_parent(nrn)


def test_load_contour_split_soma_neuron():
    nrn = utils.load_neuron(os.path.join(SWC_PATH, 'soma', 'contour_split_soma_neuron.swc'))
    nt.eq_(len(nrn.neurites), 3)
    nt.eq_(len(nrn.soma.points), 8)
    nt.eq_(nrn.soma.radius, 2.125)
    _check_neurites_have_no_parent(nrn)


def test_load_contour_split_1st_soma_neuron():
    nrn = utils.load_neuron(os.path.join(SWC_PATH, 'soma', 'contour_split_1st_soma_neuron.swc'))
    nt.eq_(len(nrn.neurites), 3)
    nt.eq_(len(nrn.soma.points), 6)
    nt.eq_(nrn.soma.radius, 2.0)
    _check_neurites_have_no_parent(nrn)


def test_load_neuromorpho_3pt_soma():
    nrn = utils.load_neuron(os.path.join(SWC_PATH, 'soma', 'three_pt_soma.swc'))
    nt.eq_(len(nrn.neurites), 4)
    nt.eq_(len(nrn.soma.points), 3)
    nt.eq_(nrn.soma.radius, 2)
    _check_neurites_have_no_parent(nrn)


@nt.raises(SomaError)
def test_load_bifurcating_soma_points_raises_SomaError():
    utils.load_neuron(os.path.join(SWC_PATH, 'soma', 'bifurcating_soma.swc'))


NRN = utils.load_neuron(FILENAMES[0])


def test_neuron_section_ids():

    # check section IDs
    for i, sec in enumerate(NRN.sections):
        nt.eq_(i, sec.id)

def test_neurites_have_no_parent():

    _check_neurites_have_no_parent(NRN)


def test_neuron_sections():
    all_nodes = set(NRN.sections)
    neurite_nodes = set(_nf.iter_sections(NRN.neurites))

    # check no duplicates
    nt.assert_true(len(all_nodes) == len(NRN.sections))

    # check all neurite tree nodes are
    # in sections attribute
    nt.assert_true(len(set(NRN.sections) - neurite_nodes) > 0)


def test_neuron_sections_are_connected():
    # check traversal by counting number of sections un trees
    for nrt in NRN.neurites:
        root_node = nrt.root_node
        nt.assert_equal(sum(1 for _ in root_node.ipreorder()),
                        sum(1 for _ in NRN.sections[root_node.id].ipreorder()))


def test_load_neuron_soma_only():

    nrn = utils.load_neuron(os.path.join(DATA_PATH, 'swc', 'Soma_origin.swc'))
    nt.eq_(len(nrn.neurites), 0)
    nt.assert_equal(nrn.name, 'Soma_origin')


@nt.raises(SomaError)
def test_load_neuron_no_soma_raises_SomaError():
    utils.load_neuron(NO_SOMA_FILE)


# TODO: decide if we want to check for this in fst.
@nt.nottest
@nt.raises(RawDataError)
def test_load_neuron_disconnected_points_raises():
    utils.load_neuron(DISCONNECTED_POINTS_FILE)


@nt.raises(RawDataError)
def test_load_neuron_missing_parents_raises():
    utils.load_neuron(MISSING_PARENTS_FILE)


# TODO: decide if we want to check for this in fst.
@nt.nottest
@nt.raises(RawDataError)
def test_load_neuron_invalid_id_sequence_raises():
    utils.load_neuron(INVALID_ID_SEQUENCE_FILE);


def test_load_neurons_directory():

    pop = utils.load_neurons(VALID_DATA_PATH)
    nt.assert_equal(len(pop.neurons), 5)
    nt.assert_equal(len(pop), 5)
    nt.assert_equal(pop.name, 'valid_set')
    for nrn in pop:
        nt.assert_true(isinstance(nrn, Neuron))


def test_load_neurons_directory_name():
    pop = utils.load_neurons(VALID_DATA_PATH, name='test123')
    nt.assert_equal(len(pop.neurons), 5)
    nt.assert_equal(len(pop), 5)
    nt.assert_equal(pop.name, 'test123')
    for nrn in pop:
        nt.assert_true(isinstance(nrn, Neuron))


def test_load_neurons_filenames():

    pop = utils.load_neurons(FILENAMES, name='test123')
    nt.assert_equal(len(pop.neurons), 3)
    nt.assert_equal(pop.name, 'test123')
    for nrn, name in zip(pop.neurons, NRN_NAMES):
        nt.assert_true(isinstance(nrn, Neuron))
        nt.assert_equal(nrn.name, name)

SWC_ORD_PATH = os.path.join(DATA_PATH, 'swc', 'ordering')
SWC_ORD_REF = utils.load_neuron(os.path.join(SWC_ORD_PATH, 'sample.swc'))


def assert_items_equal(a, b):
    nt.eq_(sorted(a), sorted(b))


def test_load_neuron_mixed_tree_swc():
    nrn_mix =  utils.load_neuron(os.path.join(SWC_ORD_PATH, 'sample_mixed_tree_sections.swc'))
    assert_items_equal(get('number_of_sections_per_neurite', nrn_mix), [5, 3])

    assert_items_equal(get('number_of_sections_per_neurite', nrn_mix),
                       get('number_of_sections_per_neurite', SWC_ORD_REF))

    assert_items_equal(get('number_of_segments', nrn_mix),
                       get('number_of_segments', SWC_ORD_REF))

    assert_items_equal(get('total_length', nrn_mix),
                       get('total_length', SWC_ORD_REF))


def test_load_neuron_section_order_break_swc():
    nrn_mix =  utils.load_neuron(os.path.join(SWC_ORD_PATH, 'sample_disordered.swc'))

    assert_items_equal(get('number_of_sections_per_neurite', nrn_mix), [5, 3])

    assert_items_equal(get('number_of_sections_per_neurite', nrn_mix),
                       get('number_of_sections_per_neurite', SWC_ORD_REF))

    assert_items_equal(get('number_of_segments', nrn_mix),
                       get('number_of_segments', SWC_ORD_REF))

    assert_items_equal(get('total_length', nrn_mix),
                       get('total_length', SWC_ORD_REF))


H5_PATH = os.path.join(DATA_PATH, 'h5', 'v1', 'ordering')
H5_ORD_REF = utils.load_neuron(os.path.join(H5_PATH, 'sample.h5'))


def test_load_neuron_mixed_tree_h5():
    nrn_mix =  utils.load_neuron(os.path.join(H5_PATH, 'sample_mixed_tree_sections.h5'))
    assert_items_equal(get('number_of_sections_per_neurite', nrn_mix), [5, 3])
    assert_items_equal(get('number_of_sections_per_neurite', nrn_mix),
                       get('number_of_sections_per_neurite', H5_ORD_REF))


def test_load_h5_trunk_points_regression():
    # regression test for issue encountered while
    # implementing PR #479, related to H5 unpacking
    # of files with non-standard soma structure.
    # See #480.
    nrn = utils.load_neuron(os.path.join(DATA_PATH, 'h5', 'v1', 'Neuron.h5'))
    nt.ok_(np.allclose(nrn.neurites[0].root_node.points[1],
                       [0., 0. , 0.1, 0.31646374, 4., 4., 3.]))

    nt.ok_(np.allclose(nrn.neurites[1].root_node.points[1],
                       [0., 0., 0.1, 1.84130445e-01, 3.0, 235., 234.]))

    nt.ok_(np.allclose(nrn.neurites[2].root_node.points[1],
                       [0., 0., 0.1, 5.62225521e-01, 3., 466, 465]))

    nt.ok_(np.allclose(nrn.neurites[3].root_node.points[1],
                       [0., 0., 0.1, 7.28555262e-01, 2., 697, 696]))


def test_load_unknown_type():
    nt.assert_raises(NeuroMError, utils.load_data, 'fake.file')


def test_NeuronLoader():
    dirpath = os.path.join(DATA_PATH, 'h5', 'v2')
    loader = utils.NeuronLoader(dirpath, file_ext='.h5', cache_size=5)
    nrn = loader.get('Neuron')
    nt.ok_(isinstance(nrn, Neuron))
    # check caching
    nt.ok_(nrn == loader.get('Neuron'))
    nt.ok_(nrn != loader.get('Neuron_2_branch'))


def test_NeuronLoader_mixed_file_extensions():
    dirpath = os.path.join(DATA_PATH, 'valid_set')
    loader = utils.NeuronLoader(dirpath)
    loader.get('Neuron')
    loader.get('Neuron_h5v1')
    nt.assert_raises(NeuroMError, loader.get, 'NoSuchNeuron')
