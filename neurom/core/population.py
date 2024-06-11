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

"""Morphology Population Classes and Functions."""
import logging
import os
from pathlib import Path

from morphio import MorphioError

import neurom
from neurom.exceptions import NeuroMError

L = logging.getLogger(__name__)


def _resolve_if_morphology_paths(files_or_objects):
    """Resolve the files in the list."""
    return [Path(os.path.abspath(f)) if isinstance(f, (Path, str)) else f for f in files_or_objects]


class Population:
    """Morphology Population Class.

    Offers an iterator over morphs within population, neurites of morphs, somas of morphs.
    It does not store the loaded morphology in memory unless the morphology has been already passed
    as loaded (instance of ``Morphology``).
    """

    def __init__(
        self,
        files,
        name='Population',
        ignored_exceptions=(),
        *,
        cache=False,
        process_subtrees=False,
    ):
        """Construct a morphology population.

        Arguments:
            files (Sequence[str|Morphology|Path]): collection of morphology files or
                paths to them or instances of ``Morphology``.
            name (str): Optional name for this Population
            ignored_exceptions (tuple): NeuroM and MorphIO exceptions that you want to ignore when
                loading morphs.
            cache (bool): whether to cache the loaded morphs in memory. If false then a morphology
                will be loaded everytime it is accessed within the population. Which is good when
                population is big. If true then all morphs will be loaded upon the construction
                and kept in memory.
            process_subtrees (bool): enable mixed tree processing if set to True

        Notes:
            symlinks in paths are not resolved.
        """
        self._ignored_exceptions = ignored_exceptions
        self.name = name

        self._files = _resolve_if_morphology_paths(files)

        self._process_subtrees = process_subtrees

        if cache:
            self._reset_cache()

    def _reset_cache(self):
        """Reset the internal cache."""
        self._files = [self._load_file(f) for f in self._files if f is not None]

    @property
    def process_subtrees(self):
        """Enable mixed tree processing if set to True."""
        return self._process_subtrees

    @process_subtrees.setter
    def process_subtrees(self, value):
        self._process_subtrees = value
        self._reset_cache()

    @property
    def morphologies(self):
        """Iterator to populations's morphologies."""
        return (n for n in self)

    @property
    def somata(self):
        """Iterator to populations's somata. Somata is the plural form of soma."""
        return (n.soma for n in self)

    @property
    def neurites(self):
        """Iterator to populations's neurites."""
        return (neurite for n in self for neurite in n.neurites)

    def _load_file(self, f):
        if isinstance(f, neurom.core.morphology.Morphology):
            new_morph = f.copy()
            new_morph.process_subtrees = self.process_subtrees
            return new_morph
        try:
            return neurom.load_morphology(f, process_subtrees=self.process_subtrees)
        except (NeuroMError, MorphioError) as e:
            if isinstance(e, self._ignored_exceptions):
                L.info('Ignoring exception "%s" for file %s', e, f.name)
            else:
                raise NeuroMError('`load_morphologies` failed') from e
        return None

    def __iter__(self):
        """Iterator to populations's morphs."""
        for f in self._files:
            m = self._load_file(f)
            if m is None:
                continue
            yield m

    def __len__(self):
        """Length of morphology collection."""
        return len(self._files)

    def __getitem__(self, idx):
        """Get morphology at index idx."""
        if idx > len(self):
            raise ValueError(
                f'no {idx} index in "{self.name}" population, max possible index is {len(self)}'
            )
        return self._load_file(self._files[idx])

    def __str__(self):
        """Return a string representation."""
        return f'Population <name: {self.name}, n_morphologies: {len(self)}>'
