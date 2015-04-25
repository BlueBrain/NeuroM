"""
Module containing the common functionality
to be used by view-plot modules.
"""
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.use('Agg')


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


def get_figure(new_fig=True, subplot=False, params={}, no_axes=False):
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

    if type(subplot) == tuple or type(subplot) == list:
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


def style_plot(fig, ax, **kwargs):

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

    title : str
        The title for the figure
        Default is "Figure".

    xlabel : str
        Label for the x-axis.
        Default is "X".

    ylabel : str
        Label for the y-axis.
        Default is "Y".

    show_plot : bool
        If True the figure is displayed.
        Default value is True.

    titlefontsize : int
        The fontsize for the title. Default is 16.

    labelfontsize : int
        The fontsize for the labels. Default is 14,

    Returns
    -------
    figure_output : figure

    """

    # Definition of title options
    no_title = kwargs.get('no_title', False)
    pretitle = kwargs.get('pretitle', '')
    posttitle = kwargs.get('posttitle', '')
    title = kwargs.get('title', 'Figure')
    titlefontsize = kwargs.get('titlefontsize', 14)
    titlearg = kwargs.get('titlearg', {})

    # Definition of label options
    no_labels = kwargs.get('no_labels', False)
    no_xlabel = kwargs.get('no_xlabel', False)
    no_ylabel = kwargs.get('no_ylabel', False)
    xlabel = kwargs.get('xlabel', 'X')
    ylabel = kwargs.get('ylabel', 'Y')
    labelfontsize = kwargs.get('labelfontsize', 14)
    xlabelarg = kwargs.get('xlabelarg', {})
    ylabelarg = kwargs.get('ylabelarg', {})

    # Definition of tick options
    no_ticks = kwargs.get('no_ticks', False)
    no_xticks = kwargs.get('no_xticks', False)
    no_yticks = kwargs.get('no_yticks', False)
    x_ticks = kwargs.get('x_ticks', [])
    y_ticks = kwargs.get('y_ticks', [])
    tickfontsize = kwargs.get('tickfontsize', 12)
    xticksarg = kwargs.get('xticksarg', {})
    yticksarg = kwargs.get('yticksarg', {})

    # Definition of legend options
    no_legend = kwargs.get('no_legend', True)
    legendarg = kwargs.get('legendarg', {})

    # Definition of limit options
    no_limits = kwargs.get('no_limits', False)
    no_xlim = kwargs.get('no_xlim', False)
    no_ylim = kwargs.get('no_ylim', False)
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)

    # Definition of global options
    no_axes = kwargs.get('no_axes', False)
    show_plot = kwargs.get('show_plot', True)
    tight = kwargs.get('tight', False)

    # Definition of save options
    prefile = kwargs.get('prefile', '')
    postfile = kwargs.get('postfile', '')
    output_path = kwargs.get('output_path', None)
    output_name = kwargs.get('output_name', 'Figure')
    output_format = kwargs.get('output_format', 'png')
    dpi = kwargs.get('dpi', 300)
    transparent = kwargs.get('transparent', False)

    pretitle, posttitle, prefile, postfile = figure_naming(pretitle, posttitle, prefile, postfile)

    if no_title:
        ax.set_title('')
    else:
        ax.set_title(pretitle + title + posttitle, fontsize=titlefontsize, **titlearg)

    if no_labels or no_xlabel:
        ax.set_xlabel('')
    else:
        ax.set_xlabel(xlabel, fontsize=labelfontsize, **xlabelarg)

    if no_labels or no_ylabel:
        ax.set_ylabel('')
    else:
        ax.set_ylabel(ylabel, fontsize=labelfontsize, **ylabelarg)

    if no_ticks or no_xticks:
        ax.xaxis.set_ticks([])
    else:
        ax.xaxis.set_ticklabels(x_ticks)
        ax.xaxis.set_tick_params(labelsize=tickfontsize, **xticksarg)
    if no_ticks or no_yticks:
        ax.yaxis.set_ticks([])
    else:
        ax.yaxis.set_ticklabels(y_ticks)
        ax.yaxis.set_tick_params(labelsize=tickfontsize, **yticksarg)

    if not no_limits and not no_xlim:
        ax.set_xlim(xlim)
    if not no_limits and not no_ylim:
        ax.set_ylim(ylim)

    if not no_legend:
        ax.legend(**legendarg)

    if no_axes:
        ax.set_frame_on(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if tight:
        plt.gcf().set_tight_layout(True)

    save_params = {'prefile': prefile,
                   'postfile': postfile,
                   'output_path': output_path,
                   'output_name': output_name,
                   'output_format': output_format,
                   'dpi': dpi,
                   'transparent': transparent}

    if output_path is not None:
        save_plot(fig=ax, **save_params)

    if not show_plot:
        plt.close()
        return (None, None)

    return fig, ax
