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

'''Simple Histogram function for multiple neurons
'''

from neurom.view import common


def histogram(neurons, feature, new_fig=True, subplot=False, **kwargs):
    '''
    Plot a histogram of the selected feature for the population of neurons.
    Plots x-axis versus y-axis on a scatter|histogram|binned values plot.

    More information about the plot and how it works.

    Parameters
    ----------
    feature : str
    The feature of interest.

    subplot : int
    The subplot identifier e. 223 for the third subplot in a 2x2
    subplot grid.

    neurons : list
        List of Neurons. Single neurons must be encapsulated in a list.

    Options
    -------
    bins : int
    Number of bins for the histogram.

    cumulative : bool
    Sets cumulative histogram on.

    subplot : bool
        Default is False, which returns a matplotlib figure object. If True,
        returns a matplotlib axis object, for use as a subplot.
    output_format : str

    Returns
    -------
    figure_output : list
        [fig|ax, figdata, figtext]
        The first item is either a figure object (if subplot is False) or an
        axis object. The second item is an object containing the data used to
        generate the figure. The final item is text used in report generation
        as a figure legend. This text needs to be manually entered in each
        figure file.

    Notes
    -----
    Any notes would go here.

    References
    ----------
    [1] Ascoli GA, Krichmar JL (2000) L-neuron: A modeling tool for the
    efficient generation and parsimonious description of dendritic morphology.
    Neurocomputing 32-33:1003-1011

    Examples
    --------
    This is an optional section giving examples of how to use the function.
    It should be formatted as follows:

    >>> function_output = my_function(param1, param2)
    function_output

    '''

    bins = kwargs.get('bins', 100)
    cumulative = kwargs.get('cumulative', False)

    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    kwargs['xlabel'] = kwargs.get('xlabel', feature)

    kwargs['ylabel'] = kwargs.get('ylabel', feature + ' fraction')

    kwargs['title'] = kwargs.get('title', feature + ' histogram')

    feature_values = [getattr(neu, 'get_' + feature)() for neu in neurons]

    neu_labels = ['neuron_id' for neu in neurons]

    ax.hist(feature_values, bins=bins, cumulative=cumulative, label=neu_labels)

    kwargs['no_legend'] = len(neu_labels) == 1

    return common.plot_style(fig=fig, ax=ax, **kwargs)
