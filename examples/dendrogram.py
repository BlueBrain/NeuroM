#!/usr/bin/env python
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

'''Moprhology Dendrogram Functions Example'''

import numpy as np
from neurom.core.tree import val_iter, Tree
from neurom.core.types import NeuriteType
from neurom.view import common
from neurom.analysis.morphmath import segment_length
from neurom.analysis.morphtree import n_segments, n_bifurcations, n_terminations

from matplotlib.collections import PolyCollection
import sys
sys.setrecursionlimit(10000)


def _vertical_segment(offsets, new_offsets, spacing, radii):
    '''Vertices fo a vertical segment
    '''
    return np.array(((new_offsets[0] - radii[0], offsets[1] + spacing[1]),
                     (new_offsets[0] - radii[1], new_offsets[1]),
                     (new_offsets[0] + radii[1], new_offsets[1]),
                     (new_offsets[0] + radii[0], offsets[1] + spacing[1])))


def _horizontal_segment(offsets, new_offsets, spacing, diameter):
    '''Vertices of a horizontal segmen
    '''
    return np.array(((offsets[0], offsets[1] + spacing[1]),
                     (new_offsets[0], offsets[1] + spacing[1]),
                     (new_offsets[0], offsets[1] + spacing[1] - diameter),
                     (offsets[0], offsets[1] + spacing[1] - diameter)))


def _spacingx(current_node, max_dims, offsets, spacing):
    '''Determine the spacing of the current node depending on the number
       of the leaves of the tree
    '''
    x_spacing = n_terminations(current_node) * spacing[0]

    if x_spacing > max_dims[0]:
        max_dims[0] = x_spacing

    return offsets[0] - x_spacing / 2.


def _generate_dendro(current_node, lines, colors, n, max_dims,
                     spacing, offsets, show_diameters=True):
    '''Recursive function for dendrogram line computations
    '''

    start_x = _spacingx(current_node, max_dims, offsets, spacing)

    radii = [0., 0.]
    # store the parent radius in order to construct polygonal segments
    # isntead of simple line segments
    radii[0] = current_node.value[3] if show_diameters else 0.

    for child in current_node.children:

        # segment length
        length = segment_length(list(val_iter((current_node, child))))

        # extract the radius of the child node. Need both radius for
        # realistic segment representation
        radii[1] = child.value[3] if show_diameters else 0.

        # number of leaves in child
        terminations = n_terminations(child)

        # horizontal spacing with respect to the number of
        # terminations
        new_offsets = (start_x + spacing[0] * terminations / 2.,
                       offsets[1] + spacing[1] * 2. + length)

        # vertical segment
        lines[n[0]] = _vertical_segment(offsets, new_offsets, spacing, radii)

        # assign segment id to color array
        colors[n[0]] = child.value[4]
        n[0] += 1

        if offsets[1] + spacing[1] * 2 + length > max_dims[1]:
            max_dims[1] = offsets[1] + spacing[1] * 2. + length

        # recursive call to self.
        _generate_dendro(child, lines, colors, n, max_dims,
                         spacing, new_offsets, show_diameters=show_diameters)

        # update the starting position for the next child
        start_x += terminations * spacing[0]

        # write the horizontal lines only for bifurcations, where the are actual horizontal lines
        # and not zero ones
        if offsets[0] != new_offsets[0]:

            # horizontal segment
            lines[n[0]] = _horizontal_segment(offsets, new_offsets, spacing, 0.)
            colors[n[0]] = current_node.value[4]
            n[0] += 1


def _create_root_soma_tree(neuron):
    ''' soma segment to represent the soma as a square of radius equal to the soma one
    '''
    soma_radius = neuron.get_soma_radius()

    soma_center = neuron.soma.center

    soma_node0 = Tree((soma_center[0] - soma_radius, soma_center[1],
                       soma_center[2], soma_radius, 1.))

    soma_node1 = Tree((soma_center[0], soma_center[1], soma_center[2], soma_radius, 1.))

    soma_node0.add_child(soma_node1)

    return soma_node0


def _dendrogram(neuron_, show_diameters=True):
    '''Main dendrogram function
    '''
    from copy import deepcopy

    # neuron is copied because otherwise the tree modifications that follow
    # will be applied to our original object
    neuron = deepcopy(neuron_)

    # total number of lines equal to the total number of segments
    # plus the number of horizontal lines (2) per bifurcation
    n_lines = sum(n_segments(neu) + n_bifurcations(neu) * 2 for neu in neuron.neurites)

    soma_tree = _create_root_soma_tree(neuron)

    n_lines += 2

    # add all the neurites to the soma treen
    for neurite in neuron.neurites:

        soma_tree.children[0].add_child(neurite)
        n_lines += 2

    # initialize the lines matrix that is required for the PolyCollection
    # and the colors one
    lines = np.zeros((n_lines, 4, 2))
    colors = np.zeros(n_lines)

    # n is used as a list in order to be static
    n = [0]

    _generate_dendro(soma_tree, lines, colors, n, [0., 0.],
                     (40., 0.), (0., 0.), show_diameters=show_diameters)

    assert n[0] == n_lines - 1

    return lines, colors


def _generate_segment_collection(tree, show_diameters):
    '''Creates quadrilateral segment collection for the input tree
    '''
    # generate positions matrix of size (n, 4, 2) for the line segments
    # and colors array for each quadrilateral shape
    positions, colors = _dendrogram(tree, show_diameters=show_diameters)

    linked_colors = []
    string_colors = []

    for val in colors:

        color_string = common.TREE_COLOR[tuple(NeuriteType)[int(val)]]

        type_string = str(tuple(NeuriteType)[int(val)]).split('.')[-1]

        linked_colors.append((color_string, type_string))

        string_colors.append(color_string)

    # generate polycollection with all the segments for the plot
    collection = PolyCollection(positions, closed=False, antialiaseds=True,
                                edgecolors=string_colors, facecolors=string_colors)

    return collection, linked_colors


def dendrogram(tree_object, show_diameters=False, new_fig=True,
               subplot=False, **kwargs):
    '''Generates the deondrogram of the input neurite

    Arguments:

        tree_object : input tree object

    Options:

        show_diameters : bool for showing segment diameters

        subplot : Default is False, which returns a matplotlib figure object. If True,
        returns a matplotlib axis object, for use as a subplot.

    Returns:

        figure_output : list
            [fig|ax, figdata, figtext]
            The first item is either a figure object (if subplot is False) or an
            axis object. The second item is an object containing the data used to
            generate the figure. The final item is text used in report generation
            as a figure legend. This text needs to be manually entered in each
            figure file.
    '''

    collection, linked_colors = _generate_segment_collection(tree_object, show_diameters)

    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    ax.add_collection(collection)
    ax.autoscale(enable=True, tight=None)

    # dummy plots for color bar labels
    for color, label in set(linked_colors):
        ax.plot((0., 0.), (0., 0.), c=color, label=label)

    # customization settings
    kwargs['xticks'] = []
    kwargs['title'] = kwargs.get('title', 'Morphology Dendrogram')
    kwargs['xlabel'] = kwargs.get('xlabel', '')
    kwargs['ylabel'] = kwargs.get('ylabel', '')
    kwargs['no_legend'] = False

    return common.plot_style(fig=fig, ax=ax, **kwargs)


def neuron_morphology_dendrogram(neuron, suptitle='Neuron Morphology'):
    '''Figure of dendrogram and morphology view
    '''
    from neurom.view import view
    import pylab as pl

    view.neuron(neuron, new_fig=True, plane='xz', subplot=121,
                title='', xlabel='', ylabel='', xticks=[], yticks=[])
    dendrogram(neuron, show_diameters=True, new_fig=False, subplot=122, rotation='right', title='')

    pl.suptitle(suptitle)
