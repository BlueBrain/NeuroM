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

'''Refactored Example to Minimize code multiplication'''

from neurom.core.tree import isegment, isection
import neurom.analysis.morphtree as mtr

from neurom.core.types import TreeType

from neurom.core.types import checkTreeType

from itertools import chain, imap


class Neurite(object):
    ''' Neurite Class
    '''
    def __init__(self, tree):
        ''' Construct Neurite object
        '''
        self._tree = tree
        self._type = mtr.get_tree_type(tree)

    def segments(self):
        '''Returns segment iterator
        '''
        return isegment(self._tree)

    def sections(self):
        '''Returns sections iterator
        '''
        return isection(self._tree)

    @property
    def type(self):
        ''' Returns neurite type
        '''
        return self._type


def i_neurites(neurites, func, neu_filter=None):
    ns = (neurites if neu_filter is None else filter(neu_filter, neurites))
    return chain(*imap(func, ns))


class Neuron(object):
    ''' Neuron Class
    '''
    def __init__(self, soma, neurites, name='Neuron'):
        '''Construct a Neuron

        Arguments:
            soma: soma object
            neurites: iterable of neurite tree structures.
            name: Optional name for this Neuron.
        '''
        self.soma = soma
        self.neurites = neurites
        self.name = name

    def segments(self, neurite_type=TreeType.all):
        '''Returns segments
        '''
        return i_neurites(self.neurites,
                          lambda n: n.segments(),
                          neu_filter=lambda t: checkTreeType(neurite_type, t.type))

    def sections(self, neurite_type=TreeType.all):
        '''Returns sections
        '''
        return i_neurites(self.neurites,
                          lambda n: n.sections(),
                          neu_filter=lambda t: checkTreeType(neurite_type, t.type))


class Population(object):
    '''Neuron Population Class'''
    def __init__(self, neurons, name='Population'):
        '''Construct a Population

        Arguments:
            neurons: iterable of neuron objects (core or ezy) .
            name: Optional name for this Population.
        '''
        self.neurons = neurons
        self.neurites = list(chain(*(neu.neurites for neu in self.neurons)))
        self.name = name

    @property
    def somata(self):
        '''Returns somata
        '''
        return (neu.soma for neu in self.neurons)

    def segments(self, neurite_type=TreeType.all):
        '''Returns segments
        '''
        return i_neurites(self.neurites,
                          lambda n: n.segments(),
                          neu_filter=lambda t: checkTreeType(neurite_type, t.type))

    def sections(self, neurite_type=TreeType.all):
        '''Returns sections
        '''
        return i_neurites(self.neurites,
                          lambda n: n.sections(),
                          neu_filter=lambda t: checkTreeType(neurite_type, t.type))

if __name__ == '__main__':
    from neurom.core.tree import Tree
    from neurom.core.neuron import make_soma

    TREE = Tree([0.0, 0.0, 0.0, 1.0, 1, 1, 2])
    T1 = TREE.add_child(Tree([0.0, 1.0, 0.0, 1.0, 1, 1, 2]))
    T2 = T1.add_child(Tree([0.0, 2.0, 0.0, 1.0, 1, 1, 2]))
    T3 = T2.add_child(Tree([0.0, 4.0, 0.0, 2.0, 1, 1, 2]))
    T4 = T3.add_child(Tree([0.0, 5.0, 0.0, 2.0, 1, 1, 2]))
    T5 = T4.add_child(Tree([2.0, 5.0, 0.0, 1.0, 1, 1, 2]))
    T6 = T4.add_child(Tree([0.0, 5.0, 2.0, 1.0, 1, 1, 2]))
    T7 = T5.add_child(Tree([3.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T8 = T7.add_child(Tree([4.0, 5.0, 0.0, 0.75, 1, 1, 2]))
    T9 = T6.add_child(Tree([0.0, 5.0, 3.0, 0.75, 1, 1, 2]))
    T10 = T9.add_child(Tree([0.0, 6.0, 3.0, 0.75, 1, 1, 2]))

    sm = make_soma([[0, 0, 0, 1, 1, 1, -1]])
    nrts = [Neurite(tr) for tr in [TREE, TREE, TREE]]
    nrn = Neuron(sm, nrts)
    pop = Population([nrn, nrn, nrn, nrn])

    print list(pop.segments())
    print
    print list(pop.neurons[0].segments())
    print
    print list(pop.neurons[0].neurites[0].segments())
