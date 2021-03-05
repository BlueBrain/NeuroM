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

"""Dendrogram helper functions and class."""
import numpy as np
from neurom import NeuriteType
from neurom.core import Neurite, Neuron
from neurom.core.dataformat import COLS
from neurom.morphmath import interval_lengths


class Dendrogram:
    """Dendrogram."""

    def __init__(self, neurom_section):
        """Dendrogram for NeuroM section tree.

        Args:
            neurom_section (Neurite|Neuron|Section): tree to build dendrogram for.
        """
        if isinstance(neurom_section, Neuron):
            self.neurite_type = NeuriteType.soma
            self.height = 1
            self.width = 1
            self.coords = self.get_coords(
                np.array([0, self.height]), np.array([.5 * self.width, .5 * self.width]))
            self.children = [Dendrogram(neurite.root_node) for neurite in neurom_section.neurites]
        else:
            if isinstance(neurom_section, Neurite):
                neurom_section = neurom_section.root_node
            lengths = interval_lengths(neurom_section.points, prepend_zero=True)
            radii = neurom_section.points[:, COLS.R]

            self.neurite_type = neurom_section.type
            self.height = np.sum(lengths)
            self.width = 2 * np.max(radii)
            self.coords = Dendrogram.get_coords(lengths, radii)
            self.children = [Dendrogram(child) for child in neurom_section.children]

    @staticmethod
    def get_coords(segment_lengths, segment_radii):
        """Coordinates of dendrogram as polygon with respect to (0, 0) origin.

        Args:
            segment_lengths: lengths of dendrogram segments
            segment_radii: radii of dendrogram segments

        Returns:
            (N,2) array of 2D x,y coordinates of Dendrogram polygon. N is the number of vertices.
        """
        y_coords = np.cumsum(segment_lengths)
        x_left_coords = -segment_radii
        x_right_coords = segment_radii
        left_coords = np.hstack((x_left_coords[:, np.newaxis], y_coords[:, np.newaxis]))
        right_coords = np.hstack((x_right_coords[:, np.newaxis], y_coords[:, np.newaxis]))
        right_coords = np.flip(right_coords, axis=0)
        return np.vstack((left_coords, right_coords))


def layout_dendrogram(dendrogram, origin):
    """Lays out dendrogram as an aesthetical pleasing tree.

    Args:
        dendrogram (Dendrogram): dendrogram
        origin (np.array): xy coordinates of layout origin

    Returns:
        Dict of positions per each dendrogram node. When placed in those positions, dendrogram nodes
        will represent a nice tree structure.
    """

    class _PositionedDendrogram:
        """Wrapper around dendrogram that allows to lay it out.

        The layout happens only in X coordinates. Children's Y coordinate is just a parent's Y
         + parent's height. Algorithm is that we calculate bounding rectangle width of each
         dendrogram's subtree. This width is a sum of all children widths calculated recursively
         in `total_width`. If no children then the width is the dendrogram's width. After the
         calculation we start to lay out. Each child gets its X coordinate as:
         parent's X + previous sibling children widths + half of this child's width.
        """
        HORIZONTAL_PADDING = 2

        def __init__(self, dendrogram):
            self.dendrogram = dendrogram
            self.children = [_PositionedDendrogram(child) for child in dendrogram.children]
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

    return _PositionedDendrogram(dendrogram).position_at(origin)


def get_size(positions):
    """Get the size of bounding rectangle that embodies positions.

    Args:
        positions (dict of Dendrogram: np.array): positions xy coordinates of dendrograms

    Returns:
        Tuple of width and height of bounding rectangle.
    """
    max_y_list = [dendrogram.height + coords[1] for dendrogram, coords in positions.items()]
    coords = np.array(list(positions.values()))
    width = np.max(coords[:, 0]) - np.min(coords[:, 0])
    height = np.max(max_y_list) - np.min(coords[:, 1])
    return width, height


def move_positions(positions, to_origin):
    """Move positions to a new origin.

    Args:
        positions (dict of Dendrogram: np.array): positions
        to_origin (np.array): where to move. np.array of (2,) shape for x,y coordindates.

    Returns:
        Moved positions.
    """
    to_origin = np.asarray(to_origin)
    return {dendrogram: position + to_origin for dendrogram, position in positions.items()}
