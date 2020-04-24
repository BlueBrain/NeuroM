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

"""Example for generating density plots."""

import pylab as plt
import numpy as np

from neurom import get as get_feat
from neurom.view import (common, view)
from neurom.core.types import NeuriteType


def extract_density(population, plane='xy', bins=100, neurite_type=NeuriteType.basal_dendrite):
    """Extracts the 2d histogram of the center
       coordinates of segments in the selected plane.
    """
    segment_midpoints = get_feat('segment_midpoints', population, neurite_type=neurite_type)
    horiz = segment_midpoints[:, 'xyz'.index(plane[0])]
    vert = segment_midpoints[:, 'xyz'.index(plane[1])]
    return np.histogram2d(np.array(horiz), np.array(vert), bins=(bins, bins))


def plot_density(population,  # pylint: disable=too-many-arguments, too-many-locals
                 bins=100, new_fig=True, subplot=111, levels=None, plane='xy',
                 colorlabel='Nodes per unit area', labelfontsize=16,
                 color_map='Reds', no_colorbar=False, threshold=0.01,
                 neurite_type=NeuriteType.basal_dendrite, **kwargs):
    """Plots the 2d histogram of the center
       coordinates of segments in the selected plane.
    """
    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    H1, xedges1, yedges1 = extract_density(population, plane=plane, bins=bins,
                                           neurite_type=neurite_type)

    mask = H1 < threshold  # mask = H1==0
    H2 = np.ma.masked_array(H1, mask)

    getattr(plt.cm, color_map).set_bad(color='white', alpha=None)

    plots = ax.contourf((xedges1[:-1] + xedges1[1:]) / 2,
                        (yedges1[:-1] + yedges1[1:]) / 2,
                        np.transpose(H2), # / np.max(H2),
                        cmap=getattr(plt.cm, color_map), levels=levels)

    if not no_colorbar:
        cbar = plt.colorbar(plots)
        cbar.ax.set_ylabel(colorlabel, fontsize=labelfontsize)

    kwargs['title'] = kwargs.get('title', '')
    kwargs['xlabel'] = kwargs.get('xlabel', plane[0])
    kwargs['ylabel'] = kwargs.get('ylabel', plane[1])

    return common.plot_style(fig=fig, ax=ax, **kwargs)


def plot_neuron_on_density(population, # pylint: disable=too-many-arguments
                           bins=100, new_fig=True, subplot=111, levels=None, plane='xy',
                           colorlabel='Nodes per unit area', labelfontsize=16,
                           color_map='Reds', no_colorbar=False, threshold=0.01,
                           neurite_type=NeuriteType.basal_dendrite, **kwargs):
    """Plots the 2d histogram of the center
       coordinates of segments in the selected plane
       and superimposes the view of the first neurite of the collection.
    """
    _, ax = common.get_figure(new_fig=new_fig)

    view.plot_tree(ax, population.neurites[0])

    return plot_density(population, plane=plane, bins=bins, new_fig=False, subplot=subplot,
                        colorlabel=colorlabel, labelfontsize=labelfontsize, levels=levels,
                        color_map=color_map, no_colorbar=no_colorbar, threshold=threshold,
                        neurite_type=neurite_type, **kwargs)
