# Copyright (c) 2015, Ecole Polytechnique Federal de Lausanne, Blue Brain Project
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

"""
Module containing the common functionality
to be used by view-plot modules.
"""
import os
import matplotlib
matplotlib.use('Agg') # noqa
import matplotlib.pyplot as plt


def figure_naming(pretitle=None, posttitle=None, prefile=None, postfile=None):
    """
    Helping function to define the strings that handle pre-post conventions
    for viewing - plotting title and saving options.

    Options
    ----------
    pretitle : str
        String to include before the general title of the figure.
        Default is None.

    posttitle : str
        String to include after the general title of the figure.
        Default is None.

    prefile : str
        String to include before the general filename of the figure.
        Default is None.

    postfile : str
        String to include after the general filename of the figure.
        Default is None.

    Returns
    -------
    string_output : string
        String to include in the figure name and title, in a suitable form.

    """
    if not pretitle:
        pretitle = ""
    else:
        pretitle = "%s -- " % pretitle

    if not posttitle:
        posttitle = ""
    else:
        posttitle = " -- %s" % posttitle

    if not prefile:
        prefile = ""
    else:
        prefile = "%s_" % prefile

    if not postfile:
        postfile = ""
    else:
        postfile = "_%s" % postfile

    return pretitle, posttitle, prefile, postfile


def get_figure(new_fig=True, subplot=False, params=None, no_axes=False):
    """
    Function to be used for viewing - plotting,
    to initialize the matplotlib figure - axes.

    Options
    ----------
    new_fig : boolean
        Defines if a new figure will be created.
        If False the current figure is returned.
        Default is True.

    subplot : subplot or False
        Defines if a subplot will be generated.
        If False the subplot is the standard 111.
        Default is False.

    params: dict
        If empty dictonary no supplementary parameters will be used.
        If dict not empty the parameters will be passed to the initialization of the axes.
        Default value is {}

    no_axes: boolean
        Defines the output of the function:
        If False the axes is returned.
        If True the figure is returned.
        Default value is False.

    Returns
    -------
    figure or axes
        Returns a figure if no_axes is True,
        or axes if no_axes is False.

    """
    if new_fig:
        fig = plt.figure()
    else:
        fig = plt.gcf()

    if no_axes:
        return fig

    if not subplot:
        subplot = 111

    if params is None:
        params = dict()

    if isinstance(subplot, (tuple, list)):
        ax = fig.add_subplot(subplot[0], subplot[1], subplot[2], **params)
    else:
        ax = fig.add_subplot(subplot, **params)

    return fig, ax


def save_plot(fig, **kwargs):

    """
    Function to be used for viewing - plotting
    to save a matplotlib figure.

    Input
    -------
    fig: matplotlib figure

    Options
    ----------
    prefile : str
        String to include before the general filename of the figure.
        Default is None.

    postfile : str
        String to include after the general filename of the figure.
        Default is None.

    output_path : str
        String to define the path to the output directory.
        Default value is './'

    output_name : str
        String to define the name of the output figure.
        Default value is 'Figure'

    output_format : str
        String to define the format of the output figure.
        Default value is 'png'

    dpi: int
        Define the DPI (Dots per Inch) of the figure.
        Default value is 300.

    transparent: boolean
        If True the saved figure will have a transparent background.
        Default value is False.

    Returns
    -------
    Generates a figure file in the selected directory.

    """

    prefile = kwargs.get('prefile', '')
    postfile = kwargs.get('postfile', '')
    output_path = kwargs.get('output_path', './')
    output_name = kwargs.get('output_name', 'Figure')
    output_format = kwargs.get('output_format', 'png')
    dpi = kwargs.get('dpi', 300)
    transparent = kwargs.get('transparent', False)

    if not os.path.exists(output_path):
        os.makedirs(output_path) # Make output directory if non-exsiting

    output = os.path.join(output_path, prefile + output_name + postfile + "." + output_format)

    plt.savefig(output, dpi=dpi, transparent=transparent)

    print 'Plot Saved:', output

    return fig


def plot_style(fig, ax, **kwargs):

    """
    Function to set the basic options of a matplotlib figure,
    to be used by viewing - plotting functions.

    Parameters
    ----------
    ax: matplotlib axes

    Options
    -------
    pretitle : str
        String to include before the general title of the figure.
        Default value is None.

    posttitle : str
        String to include after the general title of the figure.
        Default value is None.

    title : str
        The title for the figure
        Default value is "Figure".

    no_title : boolean
        Defines the presence of a title in the figure.
        If True the title will be set to an empty string "".
        Default value is False.

    title_fontsize : int
        Defines the size of the title's font.
        Default value is 14.

    title_arg : dict
        Defines the arguments that will be passsed
        into matplotlib as title arguments.
        Default value is None.

    xlabel : str
        The xlabel for the figure
        Default value is "X".

    ylabel : str
        The xlabel for the figure
        Default value is "Y".

    no_labels : boolean
        Defines the presence of the labels in the figure.
        If True the labels will be set to an empty string "".
        Default value is False.

    no_xlabel : boolean
        Defines the presence of the xlabel in the figure.
        If True the xlabel will be set to an empty string "".
        Default value is False.

    no_ylabel : boolean
        Defines the presence of the ylabel in the figure.
        If True the ylabel will be set to an empty string "".
        Default value is False.

    label_fontsize : int
        Defines the size of the labels' font.
        Default value is 14.

    xlabel_arg : dict
        Defines the arguments that will be passsed
        into matplotlib as xlabel arguments.
        Default value is None.

    ylabel_arg : dict
        Defines the arguments that will be passsed
        into matplotlib as ylabel arguments.
        Default value is None.

    no_ticks : boolean
        Defines the presence of x-y ticks in the figure.
        If True the ticks will be empty.
        Default value is False.

    no_xticks : boolean
        Defines the presence of x ticks in the figure.
        If True the ticks will be empty.
        Default value is False.

    no_yticks : boolean
        Defines the presence of y ticks in the figure.
        If True the ticks will be empty.
        Default value is False.

    xticks : list of ticks
        Defines the values of x ticks in the figure.
        Default value is None.

    yticks : list of ticks
        Defines the values of y ticks in the figure.
        Default value is None.

    tick_fontsize : int
        Defines the size of the ticks' font.
        Default value is 12.

    xticks_arg : dict
        Defines the arguments that will be passsed
        into matplotlib as xticks arguments.
        Default value is None.

    yticks_arg : dict
        Defines the arguments that will be passsed
        into matplotlib as yticks arguments.
        Default value is None.

    no_limits : boolean
        Defines the presence of plot limits in the figure.
        Default value is False.

    no_xlim : boolean
        Defines the presence of plot x-limits in the figure.
        Default value is False.

    no_ylim : boolean
        Defines the presence of plot y-limits in the figure.
        Default value is False.

    xlim: list of two floats
        Defines the min and the max values in x-axis.
        Default in None

    ylim: list of two floats
        Defines the min and the max values in y-axis.
        Default in None

    no_legend : boolean
        Defines the presence of a legend in the figure.
        If True the legend will not be included in the Figure.
        Default value is True.

    legend_arg : dict
        Defines the arguments that will be passsed
        into matplotlib as legend arguments.
        Default value is None.

    prefile : str
        String to include before the general filename of the figure.
        Default is None.

    postfile : str
        String to include after the general filename of the figure.
        Default is None.

    output_path : str
        String to define the path to the output directory.
        Default value is './'

    output_name : str
        String to define the name of the output figure.
        Default value is 'Figure'

    output_format : str
        String to define the format of the output figure.
        Default value is 'png'

    dpi: int
        Define the DPI (Dots per Inch) of the figure.
        Default value is 300.

    transparent: boolean
        If True the saved figure will have a transparent background.
        Default value is False.

    show_plot : boolean
        If True the figure is displayed.
        Default value is True.

    no_axes: boolean
        If True the labels and the frame will be set off.
        Default value is False.

    tight: boolean
        If True the set layout of matplotlib will be activated.
        Default value is False.

    Returns
    -------
    figure_output : figure, axes

    """
    # Definition of title/file naming variables
    prefile = kwargs.get('prefile', '')
    postfile = kwargs.get('postfile', '')
    pretitle = kwargs.get('pretitle', '')
    posttitle = kwargs.get('posttitle', '')

    # Definition of global options
    no_axes = kwargs.get('no_axes', False)
    show_plot = kwargs.get('show_plot', True)
    tight = kwargs.get('tight', False)

    # Definition of save options
    output_path = kwargs.get('output_path', None)

    pretitle, posttitle, prefile, postfile = figure_naming(pretitle, posttitle, prefile, postfile)

    fig, ax = plot_title(fig, ax, **kwargs)

    fig, ax = plot_labels(fig, ax, **kwargs)

    fig, ax = plot_ticks(fig, ax, **kwargs)

    fig, ax = plot_limits(fig, ax, **kwargs)

    fig, ax = plot_legend(fig, ax, **kwargs)

    if no_axes:
        ax.set_frame_on(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if tight:
        fig.set_tight_layout(True)

    if output_path is not None:
        fig = save_plot(fig=ax, **kwargs)

    if not show_plot:
        plt.close()
        return (None, None)

    return fig, ax


def plot_title(fig, ax, **kwargs):

    """
    Function that defines the title options
    of a matplotlib plot.

    Parameters
    ----------
    fig: matplotlib figure

    ax: matplotlib axes

    Options
    -------
    pretitle : str
        String to include before the general title of the figure.
        Default value is None.

    posttitle : str
        String to include after the general title of the figure.
        Default value is None.

    title : str
        The title for the figure
        Default value is "Figure".

    no_title : boolean
        Defines the presence of a title in the figure.
        If True the title will be set to an empty string "".
        Default value is False.

    title_fontsize : int
        Defines the size of the title's font.
        Default value is 14.

    title_arg : dict
        Defines the arguments that will be passsed
        into matplotlib as title arguments.
        Default value is None.

    Returns
    -------
    figure_output : figure, axes

    """

    # Definition of title options
    pretitle = kwargs.get('pretitle', '')
    posttitle = kwargs.get('posttitle', '')
    title = kwargs.get('title', 'Figure')
    no_title = kwargs.get('no_title', False)
    title_fontsize = kwargs.get('titlefontsize', 14)
    title_arg = kwargs.get('titlearg', None)

    if title_arg is None:
        title_arg = {}

    if no_title:
        ax.set_title('')
    else:
        ax.set_title(pretitle + title + posttitle,
                     fontsize=title_fontsize, **title_arg)

    return fig, ax


def plot_labels(fig, ax, **kwargs):

    """
    Function that defines the labels options
    of a matplotlib plot.

    Parameters
    ----------
    fig: matplotlib figure

    ax: matplotlib axes

    Options
    -------
    xlabel : str
        The xlabel for the figure
        Default value is "X".

    ylabel : str
        The xlabel for the figure
        Default value is "Y".

    no_labels : boolean
        Defines the presence of the labels in the figure.
        If True the labels will be set to an empty string "".
        Default value is False.

    no_xlabel : boolean
        Defines the presence of the xlabel in the figure.
        If True the xlabel will be set to an empty string "".
        Default value is False.

    no_ylabel : boolean
        Defines the presence of the ylabel in the figure.
        If True the ylabel will be set to an empty string "".
        Default value is False.

    label_fontsize : int
        Defines the size of the labels' font.
        Default value is 14.

    xlabel_arg : dict
        Defines the arguments that will be passsed
        into matplotlib as xlabel arguments.
        Default value is None.

    ylabel_arg : dict
        Defines the arguments that will be passsed
        into matplotlib as ylabel arguments.
        Default value is None.

    Returns
    -------
    figure_output : figure, axes

    """

    # Definition of label options
    xlabel = kwargs.get('xlabel', 'X')
    ylabel = kwargs.get('ylabel', 'Y')
    no_labels = kwargs.get('no_labels', False)
    no_xlabel = kwargs.get('no_xlabel', False)
    no_ylabel = kwargs.get('no_ylabel', False)
    label_fontsize = kwargs.get('labelfontsize', 14)
    xlabel_arg = kwargs.get('xlabelarg', None)
    ylabel_arg = kwargs.get('ylabelarg', None)

    if xlabel_arg is None:
        xlabel_arg = {}

    if ylabel_arg is None:
        ylabel_arg = {}

    if no_labels:
        ax.set_xlabel('')
        ax.set_ylabel('')
    else:
        if no_xlabel:
            ax.set_xlabel('')
        else:
            ax.set_xlabel(xlabel,
                          fontsize=label_fontsize,
                          **xlabel_arg)
        if no_ylabel:
            ax.set_ylabel('')
        else:
            ax.set_ylabel(ylabel,
                          fontsize=label_fontsize,
                          **ylabel_arg)

    return fig, ax


def plot_ticks(fig, ax, **kwargs):

    """
    Function that defines the labels options
    of a matplotlib plot.

    Parameters
    ----------
    fig: matplotlib figure

    ax: matplotlib axes

    Options
    -------
    no_ticks : boolean
        Defines the presence of x-y ticks in the figure.
        If True the ticks will be empty.
        Default value is False.

    no_xticks : boolean
        Defines the presence of x ticks in the figure.
        If True the ticks will be empty.
        Default value is False.

    no_yticks : boolean
        Defines the presence of y ticks in the figure.
        If True the ticks will be empty.
        Default value is False.

    xticks : list of ticks
        Defines the values of x ticks in the figure.
        Default value is None.

    yticks : list of ticks
        Defines the values of y ticks in the figure.
        Default value is None.

    tick_fontsize : int
        Defines the size of the ticks' font.
        Default value is 12.

    xticks_arg : dict
        Defines the arguments that will be passsed
        into matplotlib as xticks arguments.
        Default value is None.

    yticks_arg : dict
        Defines the arguments that will be passsed
        into matplotlib as yticks arguments.
        Default value is None.

    Returns
    -------
    figure_output : figure, axes

    """

    # Definition of tick options
    no_ticks = kwargs.get('no_ticks', False)
    no_xticks = kwargs.get('no_xticks', False)
    no_yticks = kwargs.get('no_yticks', False)
    xticks = kwargs.get('xticks', None)
    yticks = kwargs.get('yticks', None)
    tick_fontsize = kwargs.get('tickfontsize', 12)
    xticks_arg = kwargs.get('xticksarg', None)
    yticks_arg = kwargs.get('yticksarg', None)

    if xticks_arg is None:
        xticks_arg = {}

    if yticks_arg is None:
        yticks_arg = {}

    if not no_ticks:
        if xticks is not None:
            if not no_xticks:
                ax.set_xticks(xticks)
                ax.xaxis.set_tick_params(labelsize=tick_fontsize, **xticks_arg)
            else:
                ax.set_xticks([])
        if yticks is not None:
            if not no_yticks:
                ax.set_yticks(yticks)
                ax.yaxis.set_tick_params(labelsize=tick_fontsize, **xticks_arg)
            else:
                ax.set_yticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, ax


def plot_limits(fig, ax, **kwargs):

    """
    Function that defines the limit options
    of a matplotlib plot.

    Parameters
    ----------
    fig: matplotlib figure

    ax: matplotlib axes

    Options
    -------
    no_limits : boolean
        Defines the presence of plot limits in the figure.
        Default value is False.

    no_xlim : boolean
        Defines the presence of plot x-limits in the figure.
        Default value is False.

    no_ylim : boolean
        Defines the presence of plot y-limits in the figure.
        Default value is False.

    xlim: list of two floats
        Defines the min and the max values in x-axis.
        Default in None

    ylim: list of two floats
        Defines the min and the max values in y-axis.
        Default in None

    Returns
    -------
    figure_output : figure, axes

    """
    # Definition of limit options
    no_limits = kwargs.get('no_limits', False)
    no_xlim = kwargs.get('no_xlim', False)
    no_ylim = kwargs.get('no_ylim', False)
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)

    if not no_limits:
        if not no_xlim:
            ax.set_xlim(xlim)
        if not no_ylim:
            ax.set_ylim(ylim)

    return fig, ax


def plot_legend(fig, ax, **kwargs):

    """
    Function that defines the legend options
    of a matplotlib plot.

    Parameters
    ----------
    fig: matplotlib figure

    ax: matplotlib axes

    Options
    -------
    no_legend : boolean
        Defines the presence of a legend in the figure.
        If True the legend will not be included in the Figure.
        Default value is True.

    legend_arg : dict
        Defines the arguments that will be passsed
        into matplotlib as legend arguments.
        Default value is None.

    Returns
    -------
    figure_output : figure, axes

    """
    # Definition of legend options
    no_legend = kwargs.get('no_legend', True)
    legend_arg = kwargs.get('legendarg', None)

    if legend_arg is None:
        legend_arg = {}

    if not no_legend:
        ax.legend(**legend_arg)

    return fig, ax
