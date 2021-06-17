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

"""Neuron Population Classes and Functions."""
import logging

from morphio import MorphioError
import neurom
from neurom.exceptions import NeuroMError


L = logging.getLogger(__name__)


class Population:
    """Neuron Population Class.

    Offers an iterator over neurons within population, neurites of neurons, somas of neurons.
    It does not store the loaded neuron in memory unless the neuron has been already passed
    as loaded (instance of ``Neuron``).
    """
    def __init__(self, files, name='Population', ignored_exceptions=(), cache=False):
        """Construct a neuron population.

        Arguments:
            files (collections.abc.Sequence[str|Path|Neuron]): collection of neuron files or
                paths to them or instances of ``Neuron``.
            name (str): Optional name for this Population
            ignored_exceptions (tuple): NeuroM and MorphIO exceptions that you want to ignore when
                loading neurons.
            cache (bool): whether to cache the loaded neurons in memory. If false then a neuron
                will be loaded everytime it is accessed within the population. Which is good when
                population is big. If true then all neurons will be loaded upon the construction
                and kept in memory.
        """
        self._ignored_exceptions = ignored_exceptions
        self.name = name
        if cache:
            self._files = [self._load_file(f) for f in files if f is not None]
        else:
            self._files = files

    @property
    def neurons(self):
        """Iterator to populations's somas."""
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
        if isinstance(f, neurom.core.neuron.Neuron):
            return f
        try:
            return neurom.load_neuron(f)
        except (NeuroMError, MorphioError) as e:
            if isinstance(e, self._ignored_exceptions):
                L.info('Ignoring exception "%s" for file %s', e, f.name)
            else:
                raise NeuroMError('`load_neurons` failed') from e
        return None

    def __iter__(self):
        """Iterator to populations's neurons."""
        for f in self._files:
            nrn = self._load_file(f)
            if nrn is None:
                continue
            yield nrn

    def __len__(self):
        """Length of neuron collection."""
        return len(self._files)

    def __getitem__(self, idx):
        """Get neuron at index idx."""
        if idx > len(self):
            raise ValueError(
                f'no {idx} index in "{self.name}" population, max possible index is {len(self)}')
        return self._load_file(self._files[idx])

    def __str__(self):
        """Return a string representation."""
        return 'Population <name: %s, nneurons: %d>' % (self.name, len(self))
