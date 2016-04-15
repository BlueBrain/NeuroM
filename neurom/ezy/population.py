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

'''Population Class with basic analysis and plotting capabilities'''

from neurom.core.types import NeuriteType
from neurom.core.population import Population as CorePopulation


class Population(CorePopulation):
    '''Population Class

    Arguments:
        neurons: list of neurons (core or ezy)
    '''

    def __init__(self, neurons):
        super(Population, self).__init__(neurons)

    def iter_somata(self):
        '''
        Iterate over the neuron somata

            Returns:
                Iterator of neuron somata
        '''
        return iter(self.somata)

    def get_n_neurites(self, neurite_type=NeuriteType.all):
        '''Get the number of neurites of a given type in a population'''
        return sum(nrn.get_n_neurites(neurite_type=neurite_type) for nrn in self.iter_neurons())

    def iter_neurites(self):
        '''
        Iterate over the neurites

            Returns:
                Iterator of neurite tree iterators
        '''
        return iter(self.neurites)

    def iter_neurons(self):
        '''
        Iterate over the neurons in the population

            Returns:
                Iterator of neurons
        '''
        return iter(self.neurons)
