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
from neurom.analysis.dendrogram import Dendrogram
from matplotlib.collections import PolyCollection
from itertools import izip
from neurom.view import common


def _displace(rectangles, t):
    '''Displace the collection of rectangles
    '''
    n, m, _ = rectangles.shape

    for i in xrange(n):

        for j in xrange(m):

            rectangles[i, j, 0] += t[0]
            rectangles[i, j, 1] += t[1]


def _format_str(string):
    ''' string formatting
    '''
    return string.replace('TreeType.', '').replace('_', ' ').capitalize()


def _generate_collection(group, ax, ctype, colors):
    ''' Render rectangle collection
    '''
    color = common.TREE_COLOR[ctype]

    # generate segment collection
    collection = PolyCollection(group, closed=False, antialiaseds=True,
                                edgecolors=color, facecolors=color)

    # add it to the axes
    ax.add_collection(collection)

    # dummy plot for the legend
    if color not in colors:
        ax.plot((0., 0.), (0., 0.), c=color, label=_format_str(str(ctype)))
        colors.add(color)


def dendrogram(obj, new_fig=True, subplot=None, **kwargs):
    '''
    Dendrogram Viewer
    '''
    dnd = Dendrogram(obj)
    dnd.generate()

    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    displacement = 0.
    colors = set()

    n = 0
    for indices, ctype in izip(dnd.groups, dnd.types):

        # slice rectangles array for the current neurite
        group = dnd.data[indices[0]: indices[1]]

        if n > 0:
            displacement += 0.5 * (dnd.dims[n - 1][0] + dnd.dims[n][0])

        # arrange the trees without overlapping with each other
        _displace(group, (displacement, 0.))

        _generate_collection(group, ax, ctype, colors)

        n += 1

    ax.autoscale(enable=True, tight=None)

    # customization settings
    # kwargs['xticks'] = []
    kwargs['title'] = kwargs.get('title', 'Morphology Dendrogram')
    kwargs['xlabel'] = kwargs.get('xlabel', '')
    kwargs['ylabel'] = kwargs.get('ylabel', '')
    kwargs['no_legend'] = False

    return common.plot_style(fig=fig, ax=ax, **kwargs)
