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

import glob
import logging
import os
import shutil
import tempfile
import uuid
from functools import partial
from io import IOBase, open

from neurom._compat import StringType, filter
from neurom.core.population import Population
from neurom.exceptions import NeuroMError, RawDataError
from neurom.fst._core import FstNeuron
from neurom.io import neurolucida, swc
from neurom.io.datawrapper import DataWrapper

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
            try:
                return next(filter(_is_morphology_file, candidates))
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
    lsdir = (os.path.join(directory, m) for m in os.listdir(directory))
    return list(filter(_is_morphology_file, lsdir))


def get_files_by_path(path):
    '''Get a file or set of files from a file path

    Return list of files with path
    '''
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return get_morph_files(path)

    raise IOError('Invalid data path %s' % path)


def load_neuron(handle, reader=None):
    '''Build section trees from an h5 or swc file'''
    rdw = load_data(handle, reader)
    if isinstance(handle, StringType):
        name = os.path.splitext(os.path.basename(handle))[0]
    else:
        name = None
    return FstNeuron(rdw, name)


def load_neurons(neurons,
                 neuron_loader=load_neuron,
                 name=None,
                 population_class=Population,
                 ignored_exceptions=()):
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
    if isinstance(neurons, (list, tuple)):
        files = neurons
        name = name if name is not None else 'Population'
    elif isinstance(neurons, StringType):
        files = get_files_by_path(neurons)
        name = name if name is not None else os.path.basename(neurons)

    ignored_exceptions = tuple(ignored_exceptions)
    pop = []
    for f in files:
        try:
            pop.append(neuron_loader(f))
        except NeuroMError as e:
            if isinstance(e, ignored_exceptions):
                L.info('Ignoring exception "%s" for file %s',
                       e, os.path.basename(f))
                continue
            raise

    return population_class(pop, name=name)


def _get_file(handle):
    '''Returns the filename of the file to read

    If handle is a stream, a temp file is written on disk first
    and its filename is returned'''
    if not isinstance(handle, IOBase):
        return handle

    fd, temp_file = tempfile.mkstemp(str(uuid.uuid4()), prefix='neurom-')
    os.close(fd)
    with open(temp_file, 'w') as fd:
        handle.seek(0)
        shutil.copyfileobj(handle, fd)
    return temp_file


def load_data(handle, reader=None):
    '''Unpack data into a raw data wrapper'''
    if not reader:
        reader = os.path.splitext(str(handle))[1][1:].lower()

    if reader not in _READERS:
        raise NeuroMError('Do not have a loader for "%s" extension' % reader)

    filename = _get_file(handle)
    try:
        return _READERS[reader](filename)
    except Exception as e:
        L.exception('Error reading file %s, using "%s" loader', filename, reader)
        raise RawDataError('Error reading file %s:\n%s' % (filename, str(e)))


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
