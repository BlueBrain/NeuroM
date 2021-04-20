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

"""Test neurom.io.utils."""
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
from morphio import MissingParentError, RawDataError, SomaError, UnknownFileType, MorphioError
from neurom import COLS, get, load_neuron
from neurom.core import Neuron
from neurom.exceptions import NeuroMError
from neurom.io import utils
import pytest

DATA_PATH = Path(__file__).parent.parent / 'data'
SWC_PATH = DATA_PATH / 'swc'
VALID_DATA_PATH = DATA_PATH / 'valid_set'
NRN_NAMES = ('Neuron', 'Neuron_h5v1')
FILES = [SWC_PATH / f
         for f in ['Neuron.swc',
                   'Single_apical_no_soma.swc',
                   'Single_apical.swc',
                   'Single_basal.swc',
                   'Single_axon.swc',
                   'sequential_trunk_off_0_16pt.swc',
                   'sequential_trunk_off_1_16pt.swc',
                   'sequential_trunk_off_42_16pt.swc',
                   'Neuron_no_missing_ids_no_zero_segs.swc']]
FILENAMES = [VALID_DATA_PATH / f
             for f in ['Neuron.swc', 'Neuron_h5v1.h5']]
NRN = utils.load_neuron(VALID_DATA_PATH / 'Neuron.swc')
NO_SOMA_FILE = SWC_PATH / 'Single_apical_no_soma.swc'
DISCONNECTED_POINTS_FILE = SWC_PATH / 'Neuron_disconnected_components.swc'
MISSING_PARENTS_FILE = SWC_PATH / 'Neuron_missing_parents.swc'


def _mock_load_neuron(filename):
    class MockNeuron:
        def __init__(self, name):
            self.soma = 42
            self.neurites = list()
            self.name = name

    return MockNeuron(Path(filename).stem)


def _check_neurites_have_no_parent(nrn):

    for n in nrn.neurites:
        assert n.root_node.parent is None


def test_get_morph_files():
    ref = {'Neuron_slice.h5', 'Neuron.swc', 'Neuron_h5v1.h5', 'Neuron_2_branch_h5v1.h5'}

    files = set(f.name for f in utils.get_morph_files(VALID_DATA_PATH))
    assert ref == files


def test_load_neurons():
    # List of strings
    nrns = utils.load_neurons(map(str, FILES), neuron_loader=_mock_load_neuron)
    for i, nrn in enumerate(nrns):
        assert nrn.name == FILES[i].stem

    with pytest.raises(NeuroMError):
        utils.load_neurons(MISSING_PARENTS_FILE,)

    # Single string
    nrns = utils.load_neurons(str(FILES[0]), neuron_loader=_mock_load_neuron)
    assert nrns[0].name == FILES[0].stem

    # Single Path
    nrns = utils.load_neurons(FILES[0], neuron_loader=_mock_load_neuron)
    assert nrns[0].name == FILES[0].stem

    # sequence of strings
    nrns = utils.load_neurons(map(str, FILES), neuron_loader=_mock_load_neuron)
    for i, nrn in enumerate(nrns):
        assert nrn.name == FILES[i].stem

    # sequence of Path objects
    nrns = utils.load_neurons(FILES, neuron_loader=_mock_load_neuron)
    for nrn, file in zip(nrns, FILES):
        assert nrn.name == file.stem

    # string path to a directory
    nrns = utils.load_neurons(str(SWC_PATH), neuron_loader=_mock_load_neuron)
    # is subset so that if new morpho are added to SWC_PATH, the test does not break
    assert {f.stem for f in FILES}.issubset({nrn.name for nrn in nrns})

    # Path path to a directory
    nrns = utils.load_neurons(SWC_PATH, neuron_loader=_mock_load_neuron)
    # is subset so that if new morpho are added to SWC_PATH, the test does not break
    assert {f.stem for f in FILES}.issubset({nrn.name for nrn in nrns})


def test_ignore_exceptions():
    with pytest.raises(NeuroMError):
        utils.load_neurons(MISSING_PARENTS_FILE,)
    pop = utils.load_neurons((MISSING_PARENTS_FILE,), ignored_exceptions=(RawDataError,))
    assert len(pop) == 0


def test_load_neuron():
    nrn = utils.load_neuron(FILENAMES[0])
    assert isinstance(NRN, Neuron)
    assert NRN.name == 'Neuron'
    _check_neurites_have_no_parent(nrn)

    neuron_str = u""" 1 1  0  0 0 1. -1
                      2 3  0  0 0 1.  1
                      3 3  0  5 0 1.  2
                      4 3 -5  5 0 0.  3
                      5 3  6  5 0 0.  3
                      6 2  0  0 0 1.  1
                      7 2  0 -4 0 1.  6
                      8 2  6 -4 0 0.  7
                      9 2 -5 -4 0 0.  7
                     """
    utils.load_neuron(StringIO(neuron_str), reader='swc')


def test_neuron_name():
    for fn, nn in zip(FILENAMES, NRN_NAMES):
        nrn = utils.load_neuron(fn)
        assert nrn.name == nn


def test_load_bifurcating_soma_points_raises_SomaError():
    with pytest.raises(SomaError):
        utils.load_neuron(Path(SWC_PATH, 'soma', 'bifurcating_soma.swc'))


def test_load_neuromorpho_3pt_soma():
    with warnings.catch_warnings(record=True):
        nrn = utils.load_neuron(Path(SWC_PATH, 'soma', 'three_pt_soma.swc'))
    assert len(nrn.neurites) == 4
    assert len(nrn.soma.points) == 3
    assert nrn.soma.radius == 2
    _check_neurites_have_no_parent(nrn)


def test_neurites_have_no_parent():

    _check_neurites_have_no_parent(NRN)


def test_neuron_sections():
    # check no duplicates
    assert len(set(NRN.sections)) == len(list(NRN.sections))


def test_neuron_sections_are_connected():
    # check traversal by counting number of sections un trees
    for nrt in NRN.neurites:
        root_node = nrt.root_node
        assert (sum(1 for _ in root_node.ipreorder()) ==
                        sum(1 for _ in NRN.sections[root_node.id].ipreorder()))


def test_load_neuron_soma_only():

    nrn = utils.load_neuron(Path(DATA_PATH, 'swc', 'Soma_origin.swc'))
    assert len(nrn.neurites) == 0
    assert nrn.name == 'Soma_origin'


def test_load_neuron_disconnected_points_raises():
    with pytest.raises(MorphioError, match='Warning: found a disconnected neurite'):
        load_neuron(DISCONNECTED_POINTS_FILE)


def test_load_neuron_missing_parents_raises():
    with pytest.raises(MissingParentError):
        utils.load_neuron(MISSING_PARENTS_FILE)


def test_load_neurons_directory():
    pop = utils.load_neurons(VALID_DATA_PATH)
    assert len(pop.neurons) == 4
    assert len(pop) == 4
    assert pop.name == 'valid_set'
    for nrn in pop:
        assert isinstance(nrn, Neuron)


def test_load_neurons_directory_name():
    pop = utils.load_neurons(VALID_DATA_PATH, name='test123')
    assert len(pop.neurons) == 4
    assert len(pop) == 4
    assert pop.name == 'test123'
    for nrn in pop:
        assert isinstance(nrn, Neuron)


def test_load_neurons_filenames():
    pop = utils.load_neurons(FILENAMES, name='test123')
    assert len(pop.neurons) == 2
    assert pop.name == 'test123'
    for nrn, name in zip(pop.neurons, NRN_NAMES):
        assert isinstance(nrn, Neuron)
        assert nrn.name == name


SWC_ORD_PATH = Path(DATA_PATH, 'swc', 'ordering')
SWC_ORD_REF = utils.load_neuron(Path(SWC_ORD_PATH, 'sample.swc'))


def assert_items_equal(a, b):
    assert sorted(a) == sorted(b)


def test_load_neuron_mixed_tree_swc():
    nrn_mix = utils.load_neuron(Path(SWC_ORD_PATH, 'sample_mixed_tree_sections.swc'))
    assert_items_equal(get('number_of_sections_per_neurite', nrn_mix), [5, 3])

    assert_items_equal(get('number_of_sections_per_neurite', nrn_mix),
                       get('number_of_sections_per_neurite', SWC_ORD_REF))

    assert_items_equal(get('number_of_segments', nrn_mix),
                       get('number_of_segments', SWC_ORD_REF))

    assert_items_equal(get('total_length', nrn_mix),
                       get('total_length', SWC_ORD_REF))


def test_load_neuron_section_order_break_swc():
    nrn_mix = utils.load_neuron(Path(SWC_ORD_PATH, 'sample_disordered.swc'))

    assert_items_equal(get('number_of_sections_per_neurite', nrn_mix), [5, 3])

    assert_items_equal(get('number_of_sections_per_neurite', nrn_mix),
                       get('number_of_sections_per_neurite', SWC_ORD_REF))

    assert_items_equal(get('number_of_segments', nrn_mix),
                       get('number_of_segments', SWC_ORD_REF))

    assert_items_equal(get('total_length', nrn_mix),
                       get('total_length', SWC_ORD_REF))


H5_PATH = Path(DATA_PATH, 'h5', 'v1', 'ordering')
H5_ORD_REF = utils.load_neuron(Path(H5_PATH, 'sample.h5'))


def test_load_neuron_mixed_tree_h5():
    nrn_mix = utils.load_neuron(Path(H5_PATH, 'sample_mixed_tree_sections.h5'))
    assert_items_equal(get('number_of_sections_per_neurite', nrn_mix), [5, 3])
    assert_items_equal(get('number_of_sections_per_neurite', nrn_mix),
                       get('number_of_sections_per_neurite', H5_ORD_REF))


def test_load_h5_trunk_points_regression():
    # regression test for issue encountered while
    # implementing PR #479, related to H5 unpacking
    # of files with non-standard soma structure.
    # See #480.
    nrn = utils.load_neuron(Path(DATA_PATH, 'h5', 'v1', 'Neuron.h5'))
    assert np.allclose(nrn.neurites[0].root_node.points[1, COLS.XYZR],
                       [0., 0., 0.1, 0.31646374])

    assert np.allclose(nrn.neurites[1].root_node.points[1, COLS.XYZR],
                       [0., 0., 0.1, 1.84130445e-01])

    assert np.allclose(nrn.neurites[2].root_node.points[1, COLS.XYZR],
                       [0., 0., 0.1, 5.62225521e-01])

    assert np.allclose(nrn.neurites[3].root_node.points[1, COLS.XYZR],
                       [0., 0., 0.1, 7.28555262e-01])


def test_load_unknown_type():
    with pytest.raises(UnknownFileType):
        load_neuron(DATA_PATH / 'unsupported_extension.fake')


def test_NeuronLoader():
    dirpath = Path(DATA_PATH, 'h5', 'v1')
    loader = utils.NeuronLoader(dirpath, file_ext='.h5', cache_size=5)
    nrn = loader.get('Neuron')
    assert isinstance(nrn, Neuron)
    # check caching
    assert nrn == loader.get('Neuron')
    assert nrn != loader.get('Neuron_2_branch')


def test_NeuronLoader_mixed_file_extensions():
    loader = utils.NeuronLoader(VALID_DATA_PATH)
    loader.get('Neuron')
    loader.get('Neuron_h5v1')
    with pytest.raises(NeuroMError):
        loader.get('NoSuchNeuron')


def test_get_files_by_path():
    single_neurom = utils.get_files_by_path(NO_SOMA_FILE)
    assert len(single_neurom) == 1

    neuron_dir = utils.get_files_by_path(VALID_DATA_PATH)
    assert len(neuron_dir) == 4

    with pytest.raises(IOError):
        utils.get_files_by_path(Path('this/is/a/fake/path'))

def test_h5v2_raises():
    with pytest.raises(RawDataError):
        utils.load_neuron(DATA_PATH / 'h5/v2/Neuron.h5')
