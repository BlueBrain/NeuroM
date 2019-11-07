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

'''Dendrogram helper functions and class'''
from collections import namedtuple

import numpy as np

from neurom import NeuriteType
from neurom.core import Neurite
from neurom.core.dataformat import COLS


class Dendrogram:
    '''Dendrogram'''

    def __init__(self, neurom_section, dendrogram_root=None):
        '''Dendrogram for NeuroM section tree.

        Args:
            neurom_section: NeuroM section tree.
            dendrogram_root: root of dendrogram. This is a service arg, please don't set it on
            your own. It is used to track cycles in ``neurom_section``.
        '''
        if dendrogram_root is None:
            dendrogram_root = self
            dendrogram_root.processed_section_ids = []
        if neurom_section.id in dendrogram_root.processed_section_ids:
            raise ValueError('Cycled morphology {}'.format(neurom_section))
        dendrogram_root.processed_section_ids.append(neurom_section.id)

        segments = neurom_section.points
        segment_lengths = np.linalg.norm(
            np.subtract(segments[:-1, COLS.XYZ], segments[1:, COLS.XYZ]), axis=1)
        segment_radii = segments[:, COLS.R]

        self.section_id = neurom_section.id
        self.neurite_type = neurom_section.type
        self.height = np.sum(segment_lengths)
        self.width = 2 * np.max(segment_radii)
        self.coords = Dendrogram.get_coords(segment_lengths, segment_radii)
        self.children = [Dendrogram(child, dendrogram_root) for child in neurom_section.children]

    @staticmethod
    def get_coords(segment_lengths, segment_radii):
        """Coordinates of dendrogram as polygon with respect to (0, 0) origin.

        Args:
            segment_lengths: lengths of dendrogram segments
            segment_radii: radii of dendrogram segments

        Returns:
            (N,2) array of 2D x,y coordinates of Dendrogram polygon. N is the number of vertices.
        """
        y_coords = np.insert(segment_lengths, 0, 0)
        y_coords = np.cumsum(y_coords)
        x_left_coords = -segment_radii
        x_right_coords = segment_radii
        left_coords = np.hstack((x_left_coords[:, np.newaxis], y_coords[:, np.newaxis]))
        right_coords = np.hstack((x_right_coords[:, np.newaxis], y_coords[:, np.newaxis]))
        right_coords = np.flip(right_coords, axis=0)
        return np.vstack((left_coords, right_coords))


def create_dendrogram(neuron):
    '''Creates a dendrogram for neuron

    Args:
        neuron (Neurite|Neuron): Can be a Neurite or a Neuron instance.

    Returns:
        Dendrogram of ``neuron``.
    '''
    if isinstance(neuron, Neurite):
        return Dendrogram(neuron.root_node)
    SomaSection = namedtuple('NeuromSection', ['id', 'type', 'children', 'points'])
    soma_section = SomaSection(
        id=-1,
        type=NeuriteType.soma,
        children=[neurite.root_node for neurite in neuron.neurites],
        points=np.array([
            np.array([0, 0, 0, .5]),
            np.array([.1, .1, .1, .5]),
        ])
    )
    return Dendrogram(soma_section)


def layout_dendrogram(dendrogram, origin):
    '''Layouts dendrogram as an aesthetical pleasing tree.

    Args:
        dendrogram (Dendrogram): dendrogram
        origin (np.array): xy coordinates of layout origin

    Returns:
        Dict of positions per each dendrogram node. When placed in those positions, dendrogram nodes
        will represent a nice tree structure.
    '''

    class _PositionedDendrogram:
        '''Wrapper around dendrogram that allows to layout it.

        The layout happens only in X coordinates. Children's Y coordinate is just a parent's Y
         + parent's height. Algorithm is that we calculate bounding rectangle width of each
         dendrogram's subtree. This width is a sum of all children widths calculated recursively
         in `total_width`. After the calculation we start layout. Each child gets its X coordinate
         as: parent's X + previous sibling children widths + half of this child's width.
        '''
        HORIZONTAL_PADDING = 2

        def __init__(self, dendrogram):
            self.dendrogram = dendrogram
            self.children = [_PositionedDendrogram(child) for child in dendrogram.children]
            self.origin = np.empty(2)
            self.total_width = self.dendrogram.width
            if self.children:
                children_width = np.sum([child.total_width for child in self.children])
                children_width += self.HORIZONTAL_PADDING * (len(self.children) - 1)
                self.total_width = max(self.total_width, children_width)

        def position_at(self, origin):  # pylint: disable=missing-docstring
            positions = {self.dendrogram: origin}
            if self.children:
                end_point = origin + [0, self.dendrogram.height]
                left_bottom_offset = [-.5 * self.total_width, 0]
                children_origin = end_point + left_bottom_offset
                for child in self.children:
                    child_origin = children_origin + [.5 * child.total_width, 0]
                    positions.update(child.position_at(child_origin))
                    children_origin += [child.total_width + self.HORIZONTAL_PADDING, 0]
            return positions

    pos_dendrogram = _PositionedDendrogram(dendrogram)
    return pos_dendrogram.position_at(origin)


def get_size(positions):
    '''Get the size of bounding rectangle that embodies positions.

    Args:
        positions (dict of Dendrogram: np.array): positions xy coordinates of dendrograms

    Returns:
        Tuple of width and height of bounding rectangle.
    '''
    coords = np.array(list(positions.values()))
    width = np.max(coords[:, 0]) - np.min(coords[:, 0])
    max_y_list = [dendrogram.height + coords[1] for dendrogram, coords in positions.items()]
    height = np.max(max_y_list) - np.min(coords[:, 1])
    return width, height


def move_positions(positions, to_origin):
    '''Move positions to a new origin.

    Args:
        positions (dict of Dendrogram: np.array): positions
        to_origin (np.array): where to move. np.array of (2,) shape for x,y coordindates.

    Returns:
        Moved positions.
    '''
    to_origin = np.array(to_origin)
    return {dendrogram: position + to_origin for dendrogram, position in positions.items()}
