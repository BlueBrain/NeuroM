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

from neurom.view import common, view
from neurom.analysis.morphmath import segment_length
from neurom.core.tree import isegment, val_iter
from matplotlib.collections import PolyCollection
from numpy import sin, cos, pi
import pylab as pl

def segment_shape(start, end, diameter, line_type, horizontal=True):

    x0, y0 = start
    x1, y1 = end

    r = diameter / 2.

    if horizontal:

        vertex1 = ((x0, y0 + r), (x1, y1 + r))
        vertex2 = (vertex1[1], (x1, y1 - r))
        vertex3 = (vertex2[1], (x0, y0 - r))
        vertex4 = (vertex3[1], vertex1[0])

    else:

        vertex1 = ((x0 - r, y0), (x1 - r, y1 ))
        vertex2 = (vertex1[1], (x1 + r, y1))
        vertex3 = (vertex2[1], (x0 + r, y0))
        vertex4 = (vertex3[1], vertex1[0])


    return (vertex1, vertex2, vertex3, vertex4)

def get_segment_shapes(segment, segments_dict, y_length, with_diameters=False):
    '''Calculate Dendrogram Line Position. The structure of
    this drawing algorithm does not take into account directly
    the vertical segments because it stores only the ids of the
    horizontal segments which map to the ids of the nodes of the tree.
    Thus, it applies the corrections for the vertical segment dimensions
    in the horizontal one by saving virtually longer segments that
    accomodate for the diameters of the vertical segments
    '''

    start, end = segment

    start_node_id, end_node_id = start.value[-2], end.value[-2]

    # mean diameter
    d = start.value[3] + end.value[3] if with_diameters else 0.0

    segments = []

    # The parent node is connected with the child
    # line type:
    # 0 straight segment
    # 1 upward segment
    # -1 downward segment
    (x0, y0), pre_d, line_type = segments_dict[start_node_id]

    # increase horizontal dendrogram length by segment length
    x1 = x0 + segment_length(list(val_iter(segment)))

    # if the parent node is visited for the first time
    # the dendrogram segment is drawn from below. If it
    # is visited twice it is drawn from above
    y1 = y0 + y_length[start_node_id] * line_type

    # reduce the vertical y for subsequent children
    y_length[end_node_id] = y_length[start_node_id] * (1. - 0.6 * abs(line_type))

    #x0 += abs(line_type) * mean_diameter /2. 
    #x1 += abs(line_type) * mean_diameter /2. 
    #v0 = x0
    # horizontal segment
    segments.extend(segment_shape((x0, y1), (x1, y1), d, line_type, horizontal=True))

    # vertical segment
    if line_type != 0:

        #v0  = x0 + abs(line_type) * mean_diameter / 2.

        #x1 += abs(line_type) * mean_diameter

        segments.extend(segment_shape((x0, y0), (x0, y1), 0.0, line_type, horizontal=False))

    # If the segment has children, the first child will be drawn
    # from below. If no children the child will be a straight segment
    # send the first segment that starts from the ending node
    # of the current segment upwards
    # if not two children the next segment that links
    # to the current ending node will continue straight
    line_type = 1 if len(end.children) == 2 else 0



    # store the ending node id along with the data that are needed
    # for the next segment
    segments_dict[end_node_id] = [(x1, y1), d, line_type]

    # upon revisiting the starting node for the second branch
     # the direction will be opposite
    segments_dict[start_node_id][-1] = -1

    return segments


def dendro_transform(tree_object):
    '''Extract Dendrogram Lines and Diameters from tree structure
    '''
    root_id = tree_object.value[-2]

    y_length = {root_id: 500.}

    # id : [(x, y), diameter, line_type]
    segments_dict = {root_id: [(0., 0.), 0., 0.]}

    positions = []

    for seg in isegment(tree_object):

        tr_pos = get_segment_shapes(seg, segments_dict, y_length, with_diameters=True)

        positions.extend(tr_pos)

    return positions


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
    positions = dendro_transform(tree_object)
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

    #affine2D_transfrom(positions, cos(angle), -sin(angle), sin(angle), cos(angle))

    collection = PolyCollection(positions, closed=True, antialiaseds=True, edgecolors='k')

    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    ax.add_collection(collection)

    ax.autoscale(enable=True, tight=None)

    kwargs['xlim'] = [1.2 * x for x in ax.get_xlim()]
    kwargs['ylim'] = [1.2 * x for x in ax.get_ylim()]

    kwargs['title'] = kwargs.get('title', 'Morphology Dendrogram')

    kwargs['xlabel'] = xlabel

    kwargs['ylabel'] = ylabel

    return collection#common.plot_style(fig=fig, ax=ax, **kwargs)

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




    
