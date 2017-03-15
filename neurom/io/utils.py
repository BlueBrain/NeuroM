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

import logging
import os
import glob

from functools import partial
from neurom.core.population import Population
from neurom.exceptions import (RawDataError, NeuroMError)
from neurom.io.datawrapper import DataWrapper
from neurom.io import (swc, neurolucida)
from neurom.fst._core import FstNeuron
from neurom._compat import filter, StringType


L = logging.getLogger(__name__)


def _is_morphology_file(filepath):
    """ Check if `filepath` is a file with one of morphology file extensions. """
    return (
        os.path.isfile(filepath) and
        os.path.splitext(filepath)[1].lower() in ('.swc', '.h5', '.asc')
    )


class NeuronLoader(object):
    """
        Caching morphology loader.

        Arguments:
            directory: path to directory with morphology files
            file_ext: file extension to look for (if not set, will pick any of .swc|.h5|.asc)
            cache_size: size of LRU cache (if not set, no caching done)
    """
    def __init__(self, directory, file_ext=None, cache_size=None):
        self.directory = directory
        self.file_ext = file_ext
        if cache_size is not None:
            from pylru import FunctionCacheManager
            self.get = FunctionCacheManager(self.get, size=cache_size)

    def _filepath(self, name):
        """ File path to `name` morphology file. """
        if self.file_ext is None:
            candidates = glob.glob(os.path.join(self.directory, name + ".*"))
            candidates = filter(_is_morphology_file, candidates)
            try:
                return next(candidates)
            except StopIteration:
                raise NeuroMError("Can not find morphology file for '%s' " % name)
        else:
            return os.path.join(self.directory, name + self.file_ext)

    # pylint:disable=method-hidden
    def get(self, name):
        """ Get `name` morphology data. """
        return load_neuron(self._filepath(name))


def get_morph_files(directory):
    '''Get a list of all morphology files in a directory

    Returns:
        list with all files with extensions '.swc' , 'h5' or '.asc' (case insensitive)
    '''
    lsdir = [os.path.join(directory, m) for m in os.listdir(directory)]
    return list(filter(_is_morphology_file, lsdir))


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
    elif isinstance(neurons, StringType):
        files = get_morph_files(neurons)
        name = name if name is not None else os.path.basename(neurons)

    pop = population_class([neuron_loader(f) for f in files], name=name)
    return pop


def load_data(filename):
    '''Unpack data into a raw data wrapper'''
    ext = os.path.splitext(filename)[1].lower()

    if ext not in _READERS:
        raise NeuroMError('Do not have a loader for "%s" extension' % ext)

    try:
        return _READERS[ext](filename)
    except Exception:
        L.exception('Error reading file %s, using "%s" loader', filename, ext)
        raise RawDataError('Error reading file %s' % filename)


def _load_h5(filename):
    '''Delay loading of h5py until it is needed'''
    from neurom.io import hdf5
    return hdf5.read(filename,
                     remove_duplicates=False,
                     data_wrapper=DataWrapper)


_READERS = {
    '.swc': partial(swc.read,
                    data_wrapper=DataWrapper),
    '.h5': _load_h5,
    '.asc': partial(neurolucida.read,
                    data_wrapper=DataWrapper)
}
