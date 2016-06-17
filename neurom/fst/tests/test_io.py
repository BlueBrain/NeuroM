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

'''Test neurom.fst._io module loaders'''

from nose import tools as nt
import os
from neurom.fst import _io
from neurom.fst import get

_path = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(_path, '../../../test_data')
DATA_PATH = os.path.join(_path, '../../../test_data/valid_set')
FILENAMES = [os.path.join(DATA_PATH, f)
             for f in ['Neuron.swc', 'Neuron_h5v1.h5', 'Neuron_h5v2.h5']]

NRN_NAMES = ('Neuron', 'Neuron_h5v1', 'Neuron_h5v2')


def test_load_neuron():

    nrn = _io.load_neuron(FILENAMES[0])
    nt.assert_true(isinstance(nrn, _io.Neuron))
    nt.assert_equal(nrn.name, 'Neuron')


def test_neuron_name():

    for fn, nn in zip(FILENAMES, NRN_NAMES):
        nrn = _io.load_neuron(fn)
        nt.eq_(nrn.name, nn)


def test_load_neuron_soma_only():

    nrn = _io.load_neuron(os.path.join(DATA_ROOT, 'swc', 'Soma_origin.swc'))
    nt.eq_(len(nrn.neurites), 0)
    nt.assert_equal(nrn.name, 'Soma_origin')


def test_load_neurons_directory():

    pop = _io.load_neurons(DATA_PATH)
    nt.assert_equal(len(pop.neurons), 5)
    nt.assert_equal(len(pop), 5)
    nt.assert_equal(pop.name, 'valid_set')
    for nrn in pop:
        nt.assert_true(isinstance(nrn, _io.Neuron))


def test_load_neurons_directory_name():
    pop = _io.load_neurons(DATA_PATH, 'test123')
    nt.assert_equal(len(pop.neurons), 5)
    nt.assert_equal(len(pop), 5)
    nt.assert_equal(pop.name, 'test123')
    for nrn in pop:
        nt.assert_true(isinstance(nrn, _io.Neuron))


def test_load_neurons_filenames():

    pop = _io.load_neurons(FILENAMES, 'test123')
    nt.assert_equal(len(pop.neurons), 3)
    nt.assert_equal(pop.name, 'test123')
    for nrn, name in zip(pop.neurons, NRN_NAMES):
        nt.assert_true(isinstance(nrn, _io.Neuron))
        nt.assert_equal(nrn.name, name)

SWC_PATH = os.path.join(DATA_ROOT, 'swc', 'ordering')
SWC_ORD_REF = _io.load_neuron(os.path.join(SWC_PATH, 'sample.swc'))


def test_load_neuron_mixed_tree_swc():
    nrn_mix =  _io.load_neuron(os.path.join(SWC_PATH, 'sample_mixed_tree_sections.swc'))
    nt.assert_items_equal(get('number_of_sections_per_neurite', nrn_mix), [5, 3])
    nt.assert_items_equal(get('number_of_sections_per_neurite', nrn_mix),
                          get('number_of_sections_per_neurite', SWC_ORD_REF))


def test_load_neuron_section_order_break_swc():
    nrn_mix =  _io.load_neuron(os.path.join(SWC_PATH, 'sample_disordered.swc'))
    nt.assert_items_equal(get('number_of_sections_per_neurite', nrn_mix), [5, 3])
    nt.assert_items_equal(get('number_of_sections_per_neurite', nrn_mix),
                          get('number_of_sections_per_neurite', SWC_ORD_REF))


H5_PATH = os.path.join(DATA_ROOT, 'h5', 'v1', 'ordering')
H5_ORD_REF = _io.load_neuron(os.path.join(H5_PATH, 'sample.h5'))

def test_load_neuron_mixed_tree_h5():
    nrn_mix =  _io.load_neuron(os.path.join(H5_PATH, 'sample_mixed_tree_sections.h5'))
    nt.assert_items_equal(get('number_of_sections_per_neurite', nrn_mix), [5, 3])
    nt.assert_items_equal(get('number_of_sections_per_neurite', nrn_mix),
                          get('number_of_sections_per_neurite', H5_ORD_REF))
