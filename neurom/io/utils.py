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

'''Utility functions and for loading neurons'''

import os
from functools import partial
from neurom.core.population import Population
from neurom.exceptions import RawDataError
from neurom.io.datawrapper import DataWrapper
from neurom.io import swc
from neurom.io import neurolucida
from neurom.fst._core import FstNeuron


def get_morph_files(directory):
    '''Get a list of all morphology files in a directory

    Returns:
        list with all files with extensions '.swc' , 'h5' or '.asc' (case insensitive)
    '''
    lsdir = [os.path.join(directory, m) for m in os.listdir(directory)]
    return [m for m in lsdir
            if os.path.isfile(m) and
            os.path.splitext(m)[1].lower() in ('.swc', '.h5', '.asc')]


def load_neuron(filename):
    '''Build section trees from an h5 or swc file'''
    rdw = load_data(filename)
    name = os.path.splitext(os.path.basename(filename))[0]
    return FstNeuron(rdw, name)


def load_neurons(neurons,
                 neuron_loader=load_neuron,
                 name=None,
                 population_class=Population):
    '''Create a population object from all morphologies in a directory\
        of from morphologies in a list of file names

    Parameters:
        neurons: directory path or list of neuron file paths
        neuron_loader: function taking a filename and returning a neuron
        population_class: class representing populations
        name (str): optional name of population. By default 'Population' or\
            filepath basename depending on whether neurons is list or\
            directory path respectively.

    Returns:
        neuron population object

    '''
    if isinstance(neurons, list) or isinstance(neurons, tuple):
        files = neurons
        name = name if name is not None else 'Population'
    elif isinstance(neurons, str):
        files = get_morph_files(neurons)
        name = name if name is not None else os.path.basename(neurons)

    pop = population_class([neuron_loader(f) for f in files], name=name)
    return pop


def load_data(filename):
    '''Unpack data into a raw data wrapper'''
    def _clear_ext(ext):
        '''Remove extension separation and make lowercase'''
        return ext.split(os.path.extsep)[-1].lower()

    try:
        ext = os.path.splitext(filename)[1]
        return _READERS[_clear_ext(ext)](filename)
    except StandardError:
        raise RawDataError('Error reading file %s' % filename)


def _load_h5(filename):
    '''Delay loading of h5py until it is needed'''
    from neurom.io import hdf5
    return hdf5.read(filename,
                     remove_duplicates=False,
                     data_wrapper=DataWrapper)


_READERS = {
    'swc': partial(swc.read,
                   data_wrapper=DataWrapper),
    'h5': _load_h5,
    'asc': partial(neurolucida.read,
                   data_wrapper=DataWrapper)
}
