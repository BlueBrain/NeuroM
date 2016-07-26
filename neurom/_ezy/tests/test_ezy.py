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

'''Test neurom._ezy module'''

from nose import tools as nt
import os
from neurom import _ezy

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data/valid_set')
FILENAMES = [os.path.join(DATA_PATH, f)
             for f in ['Neuron.swc', 'Neuron_h5v1.h5', 'Neuron_h5v2.h5']]

def test_load_neurons_directory():

    nrns = _ezy.load_neurons(DATA_PATH)
    nt.assert_equal(len(nrns), 5)
    for nrn in nrns:
        nt.assert_true(isinstance(nrn, _ezy.Neuron))


def test_load_neurons_filenames():

    nrns = _ezy.load_neurons(FILENAMES)
    nt.assert_equal(len(nrns), 3)
    for nrn in nrns:
        nt.assert_true(isinstance(nrn, _ezy.Neuron))


def test_load_population_directory():

    pop = _ezy.load_population(DATA_PATH)
    nt.assert_equal(len(pop.neurons), 5)
    nt.assert_equal(pop.name, 'valid_set')

    pop = _ezy.load_population(DATA_PATH, 'test123')
    nt.assert_equal(len(pop.neurons), 5)
    nt.assert_equal(pop.name, 'test123')


def test_load_population_filenames():

    pop = _ezy.load_population(FILENAMES, 'test123')
    nt.assert_equal(len(pop.neurons), 3)
    nt.assert_equal(pop.name, 'test123')
