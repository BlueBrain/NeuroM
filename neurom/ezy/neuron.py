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

# TODO: enable this warning once neurite type filtering implemented
# pylint: disable=unused-argument

''' Neuron class with basic analysis and plotting capabilities. '''

from neurom.io.utils import load_neuron
from neurom.analysis.morphtree import set_tree_type
from neurom.analysis.morphtree import i_section_length
from neurom.analysis.morphtree import i_segment_length
from neurom.analysis.morphtree import n_sections
from neurom.core.dataformat import POINT_TYPE
import numpy as np
from itertools import chain


class Neurites(object):
    ''' Enum-type class holding neurite types'''

    ALL, AXON, BASAL_DENDRITE, APICAL_DENDRITE = (None,
                                                  POINT_TYPE.AXON,
                                                  POINT_TYPE.APICAL_DENDRITE,
                                                  POINT_TYPE.BASAL_DENDRITE)


class Neuron(object):
    '''Class with basic analysis and plotting functionality'''

    _VALID_NEURITES = (Neurites.ALL,)

    def __init__(self, filename, iterable_type=np.array):
        self._iterable_type = iterable_type
        self._nrn = load_neuron(filename, set_tree_type)

    def get_section_lengths(self, neurite_type=Neurites.ALL):
        '''Get an iterable containing the lengths of all sections of a given type'''
        l = [[i for i in i_section_length(t)]
             for t in self._nrn.neurite_trees]
        return [i for i in chain(*l)]

    def get_segment_lengths(self, neurite_type=Neurites.ALL):
        '''Get an iterable containing the lengths of all segments of a given type'''
        l = [[i for i in i_segment_length(t)]
             for t in self._nrn.neurite_trees]
        return [i for i in chain(*l)]

    def get_n_sections(self, neurite_type=Neurites.ALL):
        '''Get the number of sections of a given type'''
        return sum([n_sections(t) for t in self._nrn.neurite_trees])

    def get_n_sections_per_neurite(self, neurite_type=Neurites.ALL):
        '''Get an iterable with the number of sections for each neurite type'''
        return [n_sections(t) for t in self._nrn.neurite_trees]

    def get_n_neurites(self, neurite_type=Neurites.ALL):
        '''Get the number of neurites of a given type in a neuron'''
        return len(self._nrn.neurite_trees)
