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

'''Example for comparison of the same feature of multiple cells
'''
import pylab as pl
from itertools import product


def _stylize(ax, cell, feature, row, col):
    '''Stylization modifications to the plots
    '''
    if col == 0:
        ax.set_ylabel(feature)
    else:
        ax.set_yticks([])
    if row == 0 and len(cell) == 1:
        ax.set_title(cell[0].name)


def histogram(neurons, feature, ax, bins=25, normed=True, cumulative=False):
    '''
    Plot a histogram of the selected feature for the population of neurons.
    Plots x-axis versus y-axis on a scatter|histogram|binned values plot.

    Parameters :

        neurons : neuron list

        feature : str
        The feature of interest.

        bins : int
        Number of bins for the histogram.

        cumulative : bool
        Sets cumulative histogram on.

        ax : axes object
            the axes in which the plot is taking place
    '''

    # concatenate the string 'get_' with a feature to generate the respective function's name
    feature_values = [getattr(neuron, 'get_' + feature)() for neuron in neurons]
    labels = [neuron.name for neuron in neurons]
    # generate histogram
    ax.hist(feature_values, bins=bins, cumulative=cumulative,
            normed=normed, label=labels)


def plot_feature_comparison(features, cells, function=histogram, collapsible=False):
    ''' Plots the comparison of the histograms of the features of choice for
    the list of cells that is provided.

    features: list of strings

    cells : list of neuron objects

    function: the function that is applied to display the morphometrics e.g. histogram

    collapsible: if True instead of a  N x M view where N x M (features x cells), all the cells
                 collapse on the same histogram and the appropriate legend is displayed
    '''
    Nf = len(features)
    Nc = len(cells) if not collapsible else 1

    _, axes = pl.subplots(nrows=Nf, ncols=Nc, squeeze=False)

    for i, j in product(range(Nf), range(Nc)):

        ax = axes[i, j]
        cell = [cells[j]] if not collapsible else cells
        feature = features[i]

        function(cell, feature, ax)

        _stylize(ax, cell, feature, i, j)

        if collapsible:
            ax.legend(loc='best', fontsize='small')


if __name__ == '__main__':
    from neurom.ezy import load_neurons

    nrns = load_neurons('test_data/valid_set')

    FEATURES = ['segment_lengths', 'section_lengths',
                'section_radial_distances', 'local_bifurcation_angles']

    plot_feature_comparison(FEATURES, nrns, histogram, collapsible=True)

    pl.show()
