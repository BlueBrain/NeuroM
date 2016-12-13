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

'''Neuron Population Classes and Functions
'''

from itertools import chain


class Population(object):
    '''Neuron Population Class

    Features:
        - flattened collection of neurites.
        - collection of somas, neurons.
        - iterable-like iteration over neurons.
    '''
    def __init__(self, neurons, name='Population'):
        '''Construct a neuron population

        Arguments:
            neurons: iterable of neuron objects.
            name: Optional name for this Population.
        '''
        self.neurons = tuple(neurons)
        self.somata = tuple(neu.soma for neu in neurons)
        self.neurites = tuple(chain.from_iterable(neu.neurites for neu in neurons))
        self.name = name

    def __iter__(self):
        '''Iterator to populations's neurons'''
        return iter(self.neurons)

    def __len__(self):
        '''Length of neuron collection'''
        return len(self.neurons)

    def __getitem__(self, idx):
        '''Get neuron at index idx'''
        return self.neurons[idx]

    def __str__(self):
        return 'Population <name: %s, nneurons: %d>' % (self.name, len(self.neurons))
