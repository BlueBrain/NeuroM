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

from copy import deepcopy
import numpy as np
from .tree import Tree
from ..morphmath import segment_area, segment_volume, section_length
from . import NeuriteType
from neurom._compat import zip
from neurom.utils import memoize


class Section(Tree):
    '''Class representing a neurite section'''
    def __init__(self, points, section_id=None, section_type=NeuriteType.undefined):
        super(Section, self).__init__()
        self.id = section_id
        self.points = points
        self.type = section_type

    @property
    @memoize
    def length(self):
        '''Return the path length of this section.'''
        return section_length(self.points)

    @property
    @memoize
    def area(self):
        '''Return the surface area of this section.

        The area is calculated from the segments, as defined by this
        section's points
        '''
        return sum(segment_area(s) for s in zip(self.points[:-1],
                                                self.points[1:]))

    @property
    @memoize
    def volume(self):
        '''Return the volume of this section.

        The volume is calculated from the segments, as defined by this
        section's points
        '''
        return sum(segment_volume(s) for s in zip(self.points[:-1],
                                                  self.points[1:]))

    def __str__(self):
        return 'Section(id = %s, points=%s) <parent: %s, nchildren: %d>' % \
            (self.id, self.points, self.parent, len(self.children))


class Neurite(object):
    '''Class representing a neurite tree'''
    def __init__(self, root_node):
        self.root_node = root_node
        self.type = root_node.type if hasattr(root_node, 'type') else NeuriteType.undefined

    @property
    @memoize
    def points(self):
        '''Return unordered array with all the points in this neurite'''
        # add all points in a section except the first one, which is a duplicate
        _pts = [v for s in self.root_node.ipreorder() for v in s.points[1:, :4]]
        # except for the very first point, which is not a duplicate
        _pts.insert(0, self.root_node.points[0][:4])
        return np.array(_pts)

    @property
    @memoize
    def length(self):
        '''Return the total length of this neurite.

        The length is defined as the sum of lengths of the sections.
        '''
        return sum(s.length for s in self.iter_sections())

    @property
    @memoize
    def area(self):
        '''Return the surface area of this neurite.

        The area is defined as the sum of area of the sections.
        '''
        return sum(s.area for s in self.iter_sections())

    @property
    @memoize
    def volume(self):
        '''Return the volume of this neurite.

        The volume is defined as the sum of volumes of the sections.
        '''
        return sum(s.volume for s in self.iter_sections())

    def transform(self, trans):
        '''Return a copy of this neurite with a 3D transformation applied'''
        clone = deepcopy(self)
        for n in clone.iter_sections():
            n.points[:, 0:3] = trans(n.points[:, 0:3])

        return clone

    def iter_sections(self, order=Tree.ipreorder):
        '''iteration over section nodes'''
        return order(self.root_node)

    def __deepcopy__(self, memo):
        '''Deep copy of neurite object'''
        return Neurite(deepcopy(self.root_node, memo))


class Neuron(object):
    '''Class representing a simple neuron'''
    def __init__(self, soma=None, neurites=None, sections=None):
        self.soma = soma
        self.neurites = neurites
        self.sections = sections
