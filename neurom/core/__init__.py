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

''' Core functionality and data types of NeuroM '''

from itertools import ifilter, imap, chain
from .tree import Tree
from .types import NeuriteType
from ._soma import Soma, make_soma, SomaError
from ._neuron import Section, Neurite, Neuron
from .population import Population


def iter_neurites(obj, mapfun=None, filt=None):
    '''Iterator to a neurite, neuron or neuron population

    Applies optional neurite filter and mapping functions.

    Parameters:
        obj: a neurite, neuron or neuron population.
        mapfun: optional neurite mappinf function.
        filt: optional neurite filter function.

    Examples:

        Get the number of points in each neurite in a neuron population

        >>> from neurom.core import iter_neurites
        >>> n_points = [n for n in iter_neurites(pop, lambda x : len(x.points))]

        Get the number of points in each axon in a neuron population

        >>> import neurom as nm
        >>> from neurom.core import iter_neurites
        >>> filter = lambda n : n.type == nm.AXON
        >>> mapping = lambda n : len(n.points)
        >>> n_points = [n for n in iter_neurites(pop, mapping, filter)]

    '''
    neurites = ((obj,) if isinstance(obj, Neurite)
                else (obj.neurites if hasattr(obj, 'neurites') else obj))

    neurite_iter = iter(neurites) if filt is None else ifilter(filt, neurites)
    return neurite_iter if mapfun is None else imap(mapfun, neurite_iter)


def iter_sections(neurites, iterator_type=Tree.ipreorder, neurite_filter=None):
    '''Iterator to the sections in a neurite, neuron or neuron population.

    Parameters:
        neurites: neuron, population, neurite, or iterable containing neurite objects
        iterator_type: type of the iteration (ipreorder, iupstream, ibifurcation_point)
        neurite_filter: optional top level filter on properties of neurite neurite objects.

    Examples:

        Get the number of points in each section of all the axons in a neuron population

        >>> import neurom as nm
        >>> from neurom.core import ites_sections
        >>> filter = lambda n : n.type == nm.AXON
        >>> n_points = [len(s.points) for s in iter_sections(pop,  neurite_filter=filter)]

    '''
    def _mapfun(neurite):
        '''Map an iterator type to the root node of a neurite'''
        return iterator_type(neurite.root_node)

    return chain.from_iterable(imap(_mapfun, iter_neurites(neurites, filt=neurite_filter)))
