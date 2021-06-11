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

"""Functionality for styling plots."""
from pathlib import Path

import numpy as np
from matplotlib.patches import Polygon
# needed so that projection='3d' works with fig.add_subplot
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
from scipy.linalg import norm
from scipy.spatial import ConvexHull


plt = None  # refer to _get_plt()


def _get_plt():
    """Wrapper to avoid loading matplotlib.pyplot before someone has a chance to set the backend."""
    global plt  # pylint: disable=global-statement
    import matplotlib.pyplot  # pylint: disable=import-outside-toplevel
    plt = matplotlib.pyplot


def dict_if_none(arg):
    """Return an empty dict if arg is None."""
    return arg if arg is not None else {}


def figure_naming(pretitle='', posttitle='', prefile='', postfile=''):
    """Returns a formatted string with the figure name and title.

    Helper function to define the strings that handle pre-post conventions
    for viewing - plotting title and saving options.

    Args:
        pretitle(str): String to include before the general title of the figure.
        posttitle(str): String to include after the general title of the figure.
        prefile(str): String to include before the general filename of the figure.
        postfile(str): String to include after the general filename of the figure.

    Returns:
        str: String to include in the figure name and title, in a suitable form.
    """
    if pretitle:
        pretitle = "%s -- " % pretitle

    if posttitle:
        posttitle = " -- %s" % posttitle

    if prefile:
        prefile = "%s_" % prefile

    if postfile:
        postfile = "_%s" % postfile

    return pretitle, posttitle, prefile, postfile


def get_figure(new_fig=True, subplot='111', params=None):
    """Function to be used for viewing - plotting, to initialize the matplotlib figure - axes.

    Args:
        new_fig(bool): Defines if a new figure will be created, if false current figure is used
        subplot (tuple or matplolib subplot specifier string): Create axes with these parameters
        params (dict): extra options passed to add_subplot()

    Returns:
        Matplotlib Figure and Axes
    """
    _get_plt()

    if new_fig:
        fig = plt.figure()
    else:
        fig = plt.gcf()

    params = dict_if_none(params)

    if isinstance(subplot, (tuple, list)):
        ax = fig.add_subplot(*subplot, **params)
    else:
        ax = fig.add_subplot(subplot, **params)

    return fig, ax


def save_plot(fig, prefile='', postfile='', output_path='./', output_name='Figure',
              output_format='png', dpi=300, transparent=False, **_):
    """Generates a figure file in the selected directory.

    Args:
        fig: matplotlib figure
        prefile(str): Include before the general filename of the figure
        postfile(str): Included after the general filename of the figure
        output_path(str): Define the path to the output directory
        output_name(str): String to define the name of the output figure
        output_format(str): String to define the format of the output figure
        dpi(int): Define the DPI (Dots per Inch) of the figure
        transparent(bool): If True the saved figure will have a transparent background
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    fig.savefig(Path(output_path, prefile + output_name + postfile + "." + output_format),
                dpi=dpi, transparent=transparent)


def plot_style(fig, ax,  # pylint: disable=too-many-arguments, too-many-locals
               # plot_title
               pretitle='',
               title='Figure',
               posttitle='',
               title_fontsize=14,
               title_arg=None,
               # plot_labels
               label_fontsize=14,
               xlabel=None,
               xlabel_arg=None,
               ylabel=None,
               ylabel_arg=None,
               zlabel=None,
               zlabel_arg=None,
               # plot_ticks
               tick_fontsize=12,
               xticks=None,
               xticks_args=None,
               yticks=None,
               yticks_args=None,
               zticks=None,
               zticks_args=None,
               # update_plot_limits
               white_space=30,
               # plot_legend
               no_legend=True,
               legend_arg=None,
               # internal
               no_axes=False,
               aspect_ratio='equal',
               tight=False,
               **_):
    """Set the basic options of a matplotlib figure, to be used by viewing - plotting functions.

    Args:
        fig(matplotlib figure): figure
        ax(matplotlib axes, belonging to `fig`): axes

        pretitle(str): String to include before the general title of the figure
        posttitle (str): String to include after the general title of the figure
        title (str): Set the title for the figure
        title_fontsize (int): Defines the size of the title's font
        title_arg (dict): Addition arguments for matplotlib.title() call

        label_fontsize(int): Size of the labels' font
        xlabel(str): The xlabel for the figure
        xlabel_arg(dict):  Passsed into matplotlib as xlabel arguments
        ylabel(str): The ylabel for the figure
        ylabel_arg(dict):  Passsed into matplotlib as ylabel arguments
        zlabel(str): The zlabel for the figure
        zlabel_arg(dict):  Passsed into matplotlib as zlabel arguments

        tick_fontsize (int): Defines the size of the ticks' font
        xticks([list of ticks]): Defines the values of x ticks in the figure
        xticks_args(dict):  Passsed into matplotlib as xticks arguments
        yticks([list of ticks]): Defines the values of y ticks in the figure
        yticks_args(dict):  Passsed into matplotlib as yticks arguments
        zticks([list of ticks]): Defines the values of z ticks in the figure
        zticks_args(dict):  Passsed into matplotlib as zticks arguments

        white_space(float): whitespace added to surround the tight limit of the data

        no_legend (bool): Defines the presence of a legend in the figure
        legend_arg (dict): Addition arguments for matplotlib.legend() call

        no_axes(bool): If True the labels and the frame will be set off
        aspect_ratio(str): Sets aspect ratio of the figure, according to matplotlib aspect_ratio
        tight(bool): If True the tight layout of matplotlib will be activated

    Returns:
        Matplotlib figure, matplotlib axes
    """
    plot_title(ax, pretitle, title, posttitle, title_fontsize, title_arg)
    plot_labels(ax, label_fontsize, xlabel, xlabel_arg, ylabel, ylabel_arg, zlabel, zlabel_arg)
    plot_ticks(ax, tick_fontsize, xticks, xticks_args, yticks, yticks_args, zticks, zticks_args)
    update_plot_limits(ax, white_space)
    plot_legend(ax, no_legend, legend_arg)

    if no_axes:
        ax.set_frame_on(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if ax.name != '3d':
        ax.set_aspect(aspect_ratio)

    if tight:
        fig.set_tight_layout(True)


def plot_title(ax, pretitle='', title='Figure', posttitle='', title_fontsize=14, title_arg=None):
    """Set title options of a matplotlib plot.

    Args:
        ax: matplotlib axes
        pretitle(str): String to include before the general title of the figure
        posttitle (str): String to include after the general title of the figure
        title (str): Set the title for the figure
        title_fontsize (int): Defines the size of the title's font
        title_arg (dict): Addition arguments for matplotlib.title() call
    """
    current_title = ax.get_title()

    if not current_title:
        current_title = pretitle + title + posttitle

    title_arg = dict_if_none(title_arg)

    ax.set_title(current_title, fontsize=title_fontsize, **title_arg)


def plot_labels(ax, label_fontsize=14,
                xlabel=None, xlabel_arg=None,
                ylabel=None, ylabel_arg=None,
                zlabel=None, zlabel_arg=None):
    """Sets the labels options of a matplotlib plot.

    Args:
        ax: matplotlib axes
        label_fontsize(int): Size of the labels' font
        xlabel(str): The xlabel for the figure
        xlabel_arg(dict):  Passsed into matplotlib as xlabel arguments
        ylabel(str): The ylabel for the figure
        ylabel_arg(dict):  Passsed into matplotlib as ylabel arguments
        zlabel(str): The zlabel for the figure
        zlabel_arg(dict):  Passsed into matplotlib as zlabel arguments
    """
    xlabel = xlabel if xlabel is not None else ax.get_xlabel() or 'X'
    ylabel = ylabel if ylabel is not None else ax.get_ylabel() or 'Y'

    xlabel_arg = dict_if_none(xlabel_arg)
    ylabel_arg = dict_if_none(ylabel_arg)

    ax.set_xlabel(xlabel, fontsize=label_fontsize, **xlabel_arg)
    ax.set_ylabel(ylabel, fontsize=label_fontsize, **ylabel_arg)

    if hasattr(ax, 'zaxis'):
        zlabel = zlabel if zlabel is not None else ax.get_zlabel() or 'Z'
        zlabel_arg = dict_if_none(zlabel_arg)
        ax.set_zlabel(zlabel, fontsize=label_fontsize, **zlabel_arg)


def plot_ticks(ax, tick_fontsize=12,
               xticks=None, xticks_args=None,
               yticks=None, yticks_args=None,
               zticks=None, zticks_args=None):
    """Function that defines the labels options of a matplotlib plot.

    Args:
        ax: matplotlib axes
        tick_fontsize (int): Defines the size of the ticks' font
        xticks([list of ticks]): Defines the values of x ticks in the figure
        xticks_args(dict):  Passsed into matplotlib as xticks arguments
        yticks([list of ticks]): Defines the values of y ticks in the figure
        yticks_args(dict):  Passsed into matplotlib as yticks arguments
        zticks([list of ticks]): Defines the values of z ticks in the figure
        zticks_args(dict):  Passsed into matplotlib as zticks arguments
    """
    if xticks is not None:
        ax.set_xticks(xticks)
        xticks_args = dict_if_none(xticks_args)
        ax.xaxis.set_tick_params(labelsize=tick_fontsize, **xticks_args)

    if yticks is not None:
        ax.set_yticks(yticks)
        yticks_args = dict_if_none(yticks_args)
        ax.yaxis.set_tick_params(labelsize=tick_fontsize, **yticks_args)

    if zticks is not None:
        ax.set_zticks(zticks)
        zticks_args = dict_if_none(zticks_args)
        ax.zaxis.set_tick_params(labelsize=tick_fontsize, **zticks_args)


def update_plot_limits(ax, white_space):
    """Sets the limit options of a matplotlib plot.

    Args:
        ax: matplotlib axes
        white_space(float): whitespace added to surround the tight limit of the data

    Note: This relies on ax.dataLim (in 2d) and ax.[xy, zz]_dataLim being set in 3d
    """
    if hasattr(ax, 'zz_dataLim'):
        bounds = ax.xy_dataLim.bounds
        ax.set_xlim(bounds[0] - white_space, bounds[0] + bounds[2] + white_space)
        ax.set_ylim(bounds[1] - white_space, bounds[1] + bounds[3] + white_space)

        bounds = ax.zz_dataLim.bounds
        ax.set_zlim(bounds[0] - white_space, bounds[0] + bounds[2] + white_space)
    else:
        bounds = ax.dataLim.bounds
        assert not any(map(np.isinf, bounds)), 'Cannot set bounds if dataLim has infinite elements'
        ax.set_xlim(bounds[0] - white_space, bounds[0] + bounds[2] + white_space)
        ax.set_ylim(bounds[1] - white_space, bounds[1] + bounds[3] + white_space)


def plot_legend(ax, no_legend=True, legend_arg=None):
    """Function that defines the legend options of a matplotlib plot.

    Args:
        ax: matplotlib axes
        no_legend (bool): Defines the presence of a legend in the figure
        legend_arg (dict): Addition arguments for matplotlib.legend() call
    """
    legend_arg = dict_if_none(legend_arg)

    if not no_legend:
        ax.legend(**legend_arg)


_LINSPACE_COUNT = 300


def _get_normals(v):
    """Get two vectors that form a basis w/ v.

    Note: returned vectors are unit
    """
    not_v = np.array([1, 0, 0])
    if np.all(np.abs(v) == not_v):
        not_v = np.array([0, 1, 0])
    n1 = np.cross(v, not_v)
    n1 /= norm(n1)
    n2 = np.cross(v, n1)
    return n1, n2


def generate_cylindrical_points(start, end, start_radius, end_radius,
                                linspace_count=_LINSPACE_COUNT):
    """Generate a 3d mesh of a cylinder with start and end points, and varying radius.

    Based on: http://stackoverflow.com/a/32383775
    """
    v = end - start
    length = norm(v)
    v = v / length
    n1, n2 = _get_normals(v)

    # pylint: disable=unbalanced-tuple-unpacking
    l, theta = np.meshgrid(np.linspace(0, length, linspace_count),
                           np.linspace(0, 2 * np.pi, linspace_count))

    radii = np.linspace(start_radius, end_radius, linspace_count)
    rsin = np.multiply(radii, np.sin(theta))
    rcos = np.multiply(radii, np.cos(theta))

    return np.array([start[i] +
                     v[i] * l +
                     n1[i] * rsin + n2[i] * rcos
                     for i in range(3)])


def project_cylinder_onto_2d(ax, plane,
                             start, end, start_radius, end_radius,
                             color='black', alpha=1.):
    """Take cylinder defined by start/end, and project it onto the plane.

    Args:
        ax: matplotlib axes
        plane(tuple of int): where x, y, z = 0, 1, 2, so (0, 1) is the xy axis
        start(np.array): start coordinates
        end(np.array): end coordinates
        start_radius(float): start radius
        end_radius(float): end radius
        color: matplotlib color
        alpha(float): alpha value

    Note: There are probably more efficient ways of doing this: here the
    3d outline is calculated, the non-used plane coordinates are dropped, a
    tight convex hull is found, and that is used for a filled polygon
    """
    points = generate_cylindrical_points(start, end, start_radius, end_radius, 10)
    points = np.vstack([points[plane[0]].ravel(),
                        points[plane[1]].ravel()])
    points = points.T
    hull = ConvexHull(points)
    ax.add_patch(Polygon(points[hull.vertices], fill=True, color=color, alpha=alpha))


def plot_cylinder(ax, start, end, start_radius, end_radius,
                  color='black', alpha=1., linspace_count=_LINSPACE_COUNT):
    """Plot a 3d cylinder."""
    assert not np.all(start == end), 'Cylinder must have length'
    x, y, z = generate_cylindrical_points(start, end, start_radius, end_radius,
                                          linspace_count=linspace_count)
    ax.plot_surface(x, y, z, color=color, alpha=alpha)


def plot_sphere(ax, center, radius, color='black', alpha=1., linspace_count=_LINSPACE_COUNT):
    """Plots a 3d sphere, given the center and the radius."""
    u = np.linspace(0, 2 * np.pi, linspace_count)
    v = np.linspace(0, np.pi, linspace_count)
    sin_v = np.sin(v)
    x = center[0] + radius * np.outer(np.cos(u), sin_v)
    y = center[1] + radius * np.outer(np.sin(u), sin_v)
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, linewidth=0.0, color=color, alpha=alpha)
