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

"""Utility functions and for loading neurons."""

import logging
import os
import shutil
import tempfile
import uuid
from functools import lru_cache
from io import StringIO, open
from pathlib import Path

import morphio
from morphio import MorphioError
from neurom.core._neuron import Neuron
from neurom.core.population import Population
from neurom.exceptions import NeuroMError

L = logging.getLogger(__name__)


def _is_morphology_file(filepath):
    """Check if `filepath` is a file with one of morphology file extensions."""
    return filepath.is_file() and filepath.suffix.lower() in {'.swc', '.h5', '.asc'}


class NeuronLoader:
    """Caching morphology loader.

    Arguments:
        directory: path to directory with morphology files
        file_ext: file extension to look for (if not set, will pick any of .swc|.h5|.asc)
        cache_size: size of LRU cache (if not set, no caching done)
    """

    def __init__(self, directory, file_ext=None, cache_size=None):
        """Initialize a NeuronLoader object."""
        self.directory = Path(directory)
        self.file_ext = file_ext
        if cache_size is not None:
            self.get = lru_cache(maxsize=cache_size)(self.get)

    def _filepath(self, name):
        """File path to `name` morphology file."""
        if self.file_ext is None:
            candidates = self.directory.glob(name + ".*")
            try:
                return next(filter(_is_morphology_file, candidates))
            except StopIteration as e:
                raise NeuroMError("Can not find morphology file for '%s' " % name) from e
        else:
            return Path(self.directory, name + self.file_ext)

    # pylint:disable=method-hidden
    def get(self, name):
        """Get `name` morphology data."""
        return load_neuron(self._filepath(name))


def get_morph_files(directory):
    """Get a list of all morphology files in a directory.

    Returns:
        list with all files with extensions '.swc' , 'h5' or '.asc' (case insensitive)
    """
    directory = Path(directory)
    return list(filter(_is_morphology_file, directory.iterdir()))


def get_files_by_path(path):
    """Get a file or set of files from a file path.

    Return list of files with path
    """
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return get_morph_files(path)

    raise IOError('Invalid data path %s' % path)


def load_neuron(neuron, reader=None):
    """Build section trees from a neuron or a h5, swc or asc file.

    Args:
        neuron (str|Path|Neuron|morphio.Morphology|morphio.mut.Morphology): A neuron representation
            It can be:

            - a filename with the h5, swc or asc extension
            - a NeuroM Neuron object
            - a morphio mutable or immutable Morphology object
            - a stream that can be put into a io.StreamIO object. In this case, the READER argument
              must be passed with the corresponding file format (asc, swc and h5)
        reader (str): Optional, must be provided if neuron is a stream to
                      specify the file format (asc, swc, h5)

    Returns:
        A Neuron object

    Examples::

            neuron = neurom.load_neuron('my_neuron_file.h5')

            neuron = neurom.load_neuron(morphio.Morphology('my_neuron_file.h5'))

            neuron = nm.load_neuron(io.StringIO('''((Dendrite)
                                                   (3 -4 0 2)
                                                   (3 -6 0 2)
                                                   (3 -8 0 2)
                                                   (3 -10 0 2)
                                                   (
                                                     (0 -10 0 2)
                                                     (-3 -10 0 2)
                                                     |
                                                     (6 -10 0 2)
                                                     (9 -10 0 2)
                                                   )
                                                   )'''), reader='asc')
    """
    if isinstance(neuron, (Neuron, morphio.Morphology, morphio.mut.Morphology)):
        return Neuron(neuron)

    if reader:
        return Neuron(_get_file(neuron, reader))

    name = os.path.splitext(os.path.basename(neuron))[0]
    return Neuron(neuron, name)


def load_neurons(neurons,
                 neuron_loader=load_neuron,
                 name=None,
                 population_class=Population,
                 ignored_exceptions=()):
    """Create a population object.

    From all morphologies in a directory of from morphologies in a list of file names.

    Arguments:
        neurons: directory path or list of neuron file paths
        neuron_loader: function taking a filename and returning a neuron
        population_class: class representing populations
        name (str): optional name of population. By default 'Population' or\
            filepath basename depending on whether neurons is list or\
            directory path respectively.
        ignored_exceptions (tuple): NeuroM and MorphIO exceptions that you want to ignore when
            loading neurons.

    Returns:
        neuron population object
    """
    if isinstance(neurons, str):
        neurons = Path(neurons)

    if isinstance(neurons, Path):
        files = get_files_by_path(neurons)
        name = name or neurons.name
    else:
        files = neurons
        name = name or 'Population'

    ignored_exceptions = tuple(ignored_exceptions)
    pop = []
    for f in files:
        try:
            pop.append(neuron_loader(f))
        except (NeuroMError, MorphioError) as e:
            if isinstance(e, ignored_exceptions):
                L.info('Ignoring exception "%s" for file %s',
                       e, f.name)
                continue
            raise NeuroMError('`load_neurons` failed') from e

    return population_class(pop, name=name)


def _get_file(stream, extension):
    """Returns the filename of the file to read."""
    if isinstance(stream, str):
        stream = StringIO(stream)
    fd, temp_file = tempfile.mkstemp(suffix=str(uuid.uuid4()) + '.' + extension,
                                     prefix='neurom-')
    os.close(fd)
    with open(temp_file, 'w') as fd:
        stream.seek(0)
        shutil.copyfileobj(stream, fd)
    return temp_file
