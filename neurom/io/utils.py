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

"""Utility functions and for loading morphs."""

import logging
import os
import shutil
import tempfile
import uuid
from functools import lru_cache
from io import StringIO, open
from pathlib import Path

import morphio

from neurom.core.morphology import Morphology
from neurom.core.population import Population
from neurom.exceptions import NeuroMError

L = logging.getLogger(__name__)


def _is_morphology_file(filepath):
    """Check if `filepath` is a file with one of morphology file extensions."""
    return filepath.is_file() and filepath.suffix.lower() in {'.swc', '.h5', '.asc'}


class MorphLoader:
    """Caching morphology loader.

    Arguments:
        directory: path to directory with morphology files
        file_ext: file extension to look for (if not set, will pick any of .swc|.h5|.asc)
        cache_size: size of LRU cache (if not set, no caching done)
    """

    def __init__(self, directory, file_ext=None, cache_size=None):
        """Initialize a MorphLoader object."""
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
        return load_morphology(self._filepath(name))


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


def _get_file(stream, extension):
    """Returns the filename of the file to read."""
    if isinstance(stream, str):
        stream = StringIO(stream)
    fd, temp_file = tempfile.mkstemp(suffix=f'{uuid.uuid4()}.{extension}', prefix='neurom-')
    os.close(fd)
    with open(temp_file, 'w') as fd:
        stream.seek(0)
        shutil.copyfileobj(stream, fd)
    return temp_file


def load_morphology(morph, reader=None, *, mutable=None, process_subtrees=False):
    """Build section trees from a morphology or a h5, swc or asc file.

    Args:
        morph (str|Path|Morphology|morphio.Morphology|morphio.mut.Morphology): a morphology
            representation. It can be:

            - a filename with the h5, swc or asc extension
            - a NeuroM Neuron object
            - a morphio mutable or immutable Morphology object
            - a stream that can be put into a io.StreamIO object. In this case, the READER argument
              must be passed with the corresponding file format (asc, swc and h5)
        reader (str): Optional, must be provided if morphology is a stream to
                      specify the file format (asc, swc, h5)
        mutable (bool|None): Whether to enforce mutability. If None and a morphio/neurom object is
                             passed, the initial mutability will be maintained. If None and the
                             morphology is loaded, then it will be immutable by default.
        process_subtrees (bool): enable mixed tree processing if set to True

    Returns:
        A Morphology object

    Examples::

            morphology = neurom.load_morphology('my_morphology_file.h5')
            morphology = neurom.load_morphology(morphio.Morphology('my_morphology_file.h5'))
            morphology = nm.load_morphology(io.StringIO('''((Dendrite)
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
    if isinstance(morph, Morphology):
        name = morph.name
        morphio_morph = morph.to_morphio()
    elif isinstance(morph, (morphio.Morphology, morphio.mut.Morphology)):
        name = "Morphology"
        morphio_morph = morph
    else:
        filepath = _get_file(morph, reader) if reader else morph
        name = os.path.basename(filepath)
        morphio_morph = morphio.Morphology(filepath)

    # None does not modify existing mutability
    if mutable is not None:
        if mutable and isinstance(morphio_morph, morphio.Morphology):
            morphio_morph = morphio_morph.as_mutable()
        elif not mutable and isinstance(morphio_morph, morphio.mut.Morphology):
            morphio_morph = morphio_morph.as_immutable()

    return Morphology(morphio_morph, name=name, process_subtrees=process_subtrees)


def load_morphologies(
    morphs, name=None, ignored_exceptions=(), *, cache=False, process_subtrees=False
):
    """Create a population object.

    From all morphologies in a directory of from morphologies in a list of file names.

    Arguments:
        morphs(str|Path|Iterable[Path]): path to a folder or list of paths to morphology files
        name (str): optional name of population. By default 'Population' or\
            filepath basename depending on whether morphologies is list or\
            directory path respectively.
        ignored_exceptions (tuple): NeuroM and MorphIO exceptions that you want to ignore when
            loading morphologies
        cache (bool): whether to cache the loaded morphologies in memory
        process_subtrees (bool): enable mixed tree processing if set to True

    Returns:
        Population: population object
    """
    if isinstance(morphs, (str, Path)):
        files = get_files_by_path(morphs)
        name = name or Path(morphs).name
    else:
        files = morphs
        name = name or 'Population'
    return Population(
        files, name, ignored_exceptions, cache=cache, process_subtrees=process_subtrees
    )
