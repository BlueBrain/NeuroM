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

'''Neuron classes and functions'''
from neurom.core import Neuron
from neurom.core.tree import i_chain2, Tree


class PointNeuron(Neuron):
    '''Toy neuron class for testing ideas'''
    def __init__(self, soma, neurites, name='Neuron'):
        '''Construct a Neuron

        Arguments:
            soma: soma object
            neurites: iterable of neurite tree structures.
            name: Optional name for this Neuron.
        '''
        super(PointNeuron, self).__init__(soma=soma, neurites=neurites)
        self.name = name


def iter_neurites(obj, mapfun=None, filt=None):
    '''Iterator to a neurite, neuron or neuron population

    Applies optional neurite filter and element mapping functions.

    Example:
        Get the lengths of sections in a neuron and a population

        >>> from neurom import sections as sec
        >>> neuron_lengths = [l for l in iter_neurites(nrn, sec.length)]
        >>> population_lengths = [l for l in iter_neurites(pop, sec.length)]
        >>> neurite = nrn.neurites[0]
        >>> tree_lengths = [l for l in iter_neurites(neurite, sec.length)]

    '''
    #  TODO: optimize case of single neurite and move code to neurom.core.tree
    neurites = ([obj] if isinstance(obj, Tree)
                else (obj.neurites if hasattr(obj, 'neurites') else obj))
    iter_type = None if mapfun is None else mapfun.iter_type

    return i_chain2(neurites, iter_type, mapfun, filt)
