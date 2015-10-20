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
from neurom.view import common, view
from neurom.analysis.morphmath import segment_length
from neurom.analysis.morphtree import n_segments, n_bifurcations
from neurom.core.tree import isegment, val_iter, iforking_point as ifork, Tree, is_bifurcation_point, ipreorder, ileaf
from matplotlib.collections import PolyCollection
from numpy import sin, cos, pi
from copy import deepcopy
import pylab as pl
from itertools import izip

def n_leaves(tree):

    children = tree.children

    return sum(1. for _ in ileaf(tree))

def vertical_line(offsets, new_offsets, spacing, radii):

    v = np.zeros((4,2))
    #vertex1 = ((new_offsets[0] - radii[0], offsets[1] + spacing[1]), (new_offsets[0] - radii[1], new_offsets[1]))
    #vertex2 = (vertex1[1], (new_offsets[0] + radii[1], new_offsets[1]))
    #vertex3 = (vertex2[1], (new_offsets[0] + radii[0], offsets[1] + spacing[1]))
    #vertex4 = (vertex3[1], vertex1[0])

    # x,y of vertices
    v[0] = np.array((new_offsets[0] - radii[0], offsets[1] + spacing[1]))
    v[1] = np.array((new_offsets[0] - radii[1], new_offsets[1]))
    v[2] = np.array((new_offsets[0] + radii[1], new_offsets[1]))
    v[3] = np.array((new_offsets[0] + radii[0], offsets[1] + spacing[1]))

    return v

def horizontal_line(offsets, new_offsets, spacing, diameter):

    #vertex1 = ((offsets[0], offsets[1] + spacing[1]), (new_offsets[0], offsets[1] + spacing[1]))

    v = np.zeros((4,2))
    # x coordinates
    v[0] = np.array((offsets[0], offsets[1] + spacing[1]))
    v[1] = np.array((new_offsets[0], offsets[1] + spacing[1]))
    v[2] = np.array((new_offsets[0], offsets[1] + spacing[1] - diameter))
    v[3] = np.array((offsets[0], offsets[1] + spacing[1] - diameter))

    return v
    #return np.array(vertex1, vertex1, vertex1, vertex1)


def _generate_dendro(current_node, lines, n, max_dims, spacing, off_x, off_y, show_diameters=True):

    max_terminations = n_leaves(current_node)

    # determine the spacing of the current node depending on the number
    # of the leaves of the tree
    x_spacing = max_terminations * spacing[0]

    start_x = off_x - x_spacing / 2.

    if (x_spacing > max_dims[0]): max_dims[0] = x_spacing

    # store the parent radius in order to construct polygonal segments
    # isntead of simple line segments
    r_parent = current_node.value[3] if show_diameters else 0.

    for child in current_node.children:

        # segment length
        length = segment_length(list(val_iter((current_node, child))))

        # extract the radius of the child node. Need both radius for
        # realistic segment representation
        r_child = child.value[3] if show_diameters else 0.

        # number of leaves in child
        terminations = n_leaves(child)

        # horizontal spacing with respect to the number of 
        new_off_x = start_x + spacing[0] * terminations / 2.
        new_off_y = off_y + spacing[1] * 2. + length 

        # vertical segment
        lines[n] = vertical_line((off_x, off_y), (new_off_x, new_off_y), spacing, (r_parent, r_child))

        if off_y + spacing[1] * 2 + length > max_dims[1]:
            max_dims[1] = off_y + spacing[1] * 2. + length

        # recursive call to self. n must be outputed in order to be maintain
        # its actual value
        n,_ = _generate_dendro(child, lines, n + 1, max_dims, spacing, new_off_x, new_off_y, show_diameters=show_diameters)

        # update the starting position for the next child
        start_x += terminations * spacing[0]

        # write the horizontal lines only for bifurcations, where the are actual horizontal lines
        # and not zero ones
        if off_x != new_off_x and off_y != new_off_y:

            # horizontal segment
            lines[n] = horizontal_line((off_x, off_y), (new_off_x, new_off_y), spacing, 2. * r_parent)

            n += 1

    return n, spacing


def _dendrogram(neurites, show_diameters=True, transform=None):

    # total number of lines equal to the total number of segments
    # plus the number of horizontal lines (2) per bifurcation
    total_lines = sum( n_segments(neu) + n_bifurcations(neu) * 2 for neu in neurites)

    print total_lines
    

    spacing = (20., 0.)

    lines = np.zeros((total_lines, 4, 2))

    for neurite in neurites:

        max_dims = [0., 0.]
        n, spacing = _generate_dendro(neurite, lines, 0, max_dims, spacing, 0., 0., show_diameters=show_diameters)

    return lines


    


"""


def affine2D_transfrom(pos, a, b, c, d):
    '''Affine 2D transformation for the positions of the line collection
    '''

    for i, elements in enumerate(pos):
        for j, _ in enumerate(elements):

            x = pos[i][j][0]
            y = pos[i][j][1]

            x_prime = a * x + b * y
            y_prime = c * x + d * y

            pos[i][j] = (x_prime, y_prime)
"""

def dendrogram(tree_object, show_diameters=False, new_fig=True,
               subplot=False, rotation='right', **kwargs):
    '''Generates the deondrogram of the input neurite

    Arguments:

        tree_object : input tree object

    Options:

        show_diameters : bool for showing mean segment diameters

        subplot : Default is False, which returns a matplotlib figure object. If True,
        returns a matplotlib axis object, for use as a subplot.

        rotation : angle in degrees of rotation of the dendrogram.

    Returns:

        figure_output : list
            [fig|ax, figdata, figtext]
            The first item is either a figure object (if subplot is False) or an
            axis object. The second item is an object containing the data used to
            generate the figure. The final item is text used in report generation
            as a figure legend. This text needs to be manually entered in each
            figure file.
    '''
    # get line segment positions and respective diameters
    #positions = dendro_transform(tree_object)
    #min_l = 1.0
    #max_l = 100.0
    #linewidths = [ (w - min_l) * (max_l - min_l) for w in linewidths ] if show_diameters else 1.0

    xlabel = kwargs.get('xlabel', 'Length (um)')
    ylabel = kwargs.get('ylabel', '')

    if rotation == 'left':

        angle = pi

    elif rotation == 'up':

        angle = pi / 2.

        xlabel, ylabel = ylabel, xlabel

    elif rotation == 'down':

        angle = - pi / 2.

        xlabel, ylabel = ylabel, xlabel

    else:

        angle = 0.

    positions = _dendrogram(tree_object, show_diameters=show_diameters)

    #affine2D_transfrom(positions, cos(angle), -sin(angle), sin(angle), cos(angle))

    collection = PolyCollection(positions, closed=False, antialiaseds=True, edgecolors='k', facecolors='k')

    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    ax.add_collection(collection)

    ax.autoscale(enable=True, tight=None)

    kwargs['xlim'] = [1.2 * x for x in ax.get_xlim()]
    kwargs['ylim'] = [1.2 * x for x in ax.get_ylim()]

    kwargs['title'] = kwargs.get('title', 'Morphology Dendrogram')

    kwargs['xlabel'] = xlabel

    kwargs['ylabel'] = ylabel

    return common.plot_style(fig=fig, ax=ax, **kwargs)

def neurites_figure(neurites, suptitle='Neuron Morphology', **kwargs):

    fig = pl.figure()

    fig.suptitle(suptitle)

    n = len(neurites)

    for i, neurite in enumerate(neurites):

        sub1 = (n, 2, 2*i + 1)
        sub2 = (n, 2, 2*i + 2)

        xlabel = '' if i < n - 1 else 'Length (um)'

        dendrogram(neurite, show_diameters=True, new_fig=False, subplot=sub1, rotation='right', title='', xlabel=xlabel,       xticks=[], yticks=[])
        
        view.tree(neurite, new_fig=False, plane='xz', subplot=sub2, title='', xlabel=xlabel, ylabel='', xticks=[],yticks=[])




    
