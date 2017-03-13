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
'''
Python module of NeuroM to visualize morphologies
'''
from . import common

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from neurom import NeuriteType, geom

from neurom.core import iter_segments
from neurom.core._soma import SomaCylinders
from neurom.core.dataformat import COLS
from neurom.morphmath import segment_radius
from neurom.view._dendrogram import Dendrogram
from neurom._compat import zip


# XXX(mgevaert)
# Document linewidth/diameter_scale interplay
#
#        diameter_scale: float \
#            Defines the scale factor that will be multiplied \
#            with the diameter to define the width of the tree line. \
#            Default value is 1.
#        color: str or None \
#            Defines the color of the tree. \
#            If None the default values will be used, \
#            depending on the type of tree: \
#            Basal dendrite: "red" \
#            Axon : "blue" \
#            Apical dendrite: "purple" \
#            Undefined tree: "black" \
#            Default value is None.
#        linewidth: float \
#            Defines the linewidth of the tree, \
#            if diameter is set to False. \
#            Default value is 1.2.
#        alpha: float \
#            Defines the transparency of the tree. \
#            0.0 transparent through 1.0 opaque. \
#            Default value is 0.8.


DEFAULT_PARAMS = '''        new_fig: boolean \
            Defines if the tree will be plotted \
            in the current figure (False) \
            or in a new figure (True) \
            Default value is True.
        subplot: matplotlib subplot value or False \
            If False the default subplot 111 will be used. \
            For any other value a matplotlib subplot \
            will be generated. \
            Default value is False.

''' + common.PLOT_STYLE_PARAMS


_LINEWIDTH = 1.2
_ALPHA = 0.8
_DIAMETER_SCALE = 1.0


def _plane2col(plane):
    '''take a string like 'xy', and return the indices from COLS.*'''
    planes = ('xy', 'yx', 'xz', 'zx', 'yz', 'zy')
    assert plane in planes, 'No such plane found! Please select one of: ' + str(planes)
    return (getattr(COLS, plane[0].capitalize()),
            getattr(COLS, plane[1].capitalize()), )


def _get_linewidth(tr, linewidth, diameter_scale):
    '''calculate the desired linewidth based on tree contents, and parameters'''
    if diameter_scale is not None and tr:
        linewidth = [2 * segment_radius(s) * diameter_scale
                     for s in iter_segments(tr)]
    return linewidth


def tree(ax, tr, plane='xy',
         diameter_scale=_DIAMETER_SCALE, linewidth=_LINEWIDTH,
         color=None, alpha=_ALPHA,
         **_):
    '''
    Generates a 2d figure of the tree's segments. \
    If the tree contains one single point the plot will be empty \
    since no segments can be constructed.

    Parameters:
        tr: Tree \
            neurom.Tree object
        plane: str \
            Accepted values: Any pair of of xyz \
            Default value is 'xy'.
    '''
    plane0, plane1 = _plane2col(plane)
    segs = [((s[0][plane0], s[0][plane1]),
             (s[1][plane0], s[1][plane1]))
            for s in iter_segments(tr)]

    linewidth = _get_linewidth(tr, diameter_scale=diameter_scale, linewidth=linewidth)
    color = common.get_color(color, tr.type)
    collection = LineCollection(segs, color=color, linewidth=linewidth, alpha=alpha)

    ax.add_collection(collection)


def soma(ax, sm, plane='xy',
         soma_outline=True,
         linewidth=_LINEWIDTH,
         color=None, alpha=_ALPHA,
         **_):
    '''
    Generates a 2d figure of the soma.

    Parameters:
        soma: neurom.Soma object
        plane: str \
            Accepted values: Any pair of of xyz \
            Default value is 'xy'.
    '''
    plane0, plane1 = _plane2col(plane)
    color = common.get_color(color, tree_type=NeuriteType.soma)

    if isinstance(sm, SomaCylinders):
        plane0, plane1 = _plane2col(plane)
        for start, end in zip(sm.points, sm.points[1:]):
            common.project_cylinder_onto_2d(ax, (plane0, plane1),
                                            start=start[COLS.XYZ], end=end[COLS.XYZ],
                                            start_radius=start[COLS.R], end_radius=end[COLS.R],
                                            color=treecolor, alpha=alpha)
    else:
        if soma_outline:
            ax.add_artist(Circle(sm.center, sm.radius, color=color, alpha=alpha))
        else:
            plane0, plane1 = _plane2col(plane)
            points = [(p[plane0], p[plane1]) for p in sm.iter()]

            if points:
                points.append(points[0])  # close the loop
                common.plt.plot(points, color=color, alpha=alpha, linewidth=linewidth)

    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])


def neuron(ax, nrn, plane='xy',
           soma_outline=True,
           diameter_scale=_DIAMETER_SCALE, linewidth=_LINEWIDTH,
           color=None, alpha=_ALPHA,
           **_):
    '''Generates a 2d figure of the neuron, that contains a soma and a list of trees

    Parameters:
        neuron: Neuron neurom.Neuron object
        plane: str \
            Accepted values: Any pair of of xyz Default value is 'xy'.
    '''
    soma(ax, nrn.soma, plane=plane, soma_outline=soma_outline, linewidth=linewidth,
         color=color, alpha=alpha)

    for neurite in nrn.neurites:
        tree(ax, neurite, plane=plane, diameter_scale=diameter_scale, linewidth=linewidth,
             color=color, alpha=alpha)

    ax.set_title(nrn.name)
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])


def tree3d(ax, tr,
           diameter_scale=_DIAMETER_SCALE, linewidth=_LINEWIDTH,
           color=None, alpha=_ALPHA,
           **_):
    '''Generates a figure of the tree in 3d.

    If the tree contains one single point the plot will be empty \
    since no segments can be constructed.

    Parameters:
        tr: Tree neurom.Tree object
    '''
    segs = [(s[0][COLS.XYZ], s[1][COLS.XYZ]) for s in iter_segments(tr)]

    linewidth = _get_linewidth(tr, diameter_scale=diameter_scale, linewidth=linewidth)
    color = common.get_color(color, tr.type),

    collection = Line3DCollection(segs, color=color, linewidth=linewidth, alpha=alpha)
    ax.add_collection3d(collection)

    # unlike w/ 2d Axes, the dataLim isn't set by collections, so it has to be updated manually
    min_bounding_box, max_bounding_box = geom.bounding_box(tr)
    xy_bounds = np.vstack((min_bounding_box[:COLS.Z],
                           max_bounding_box[:COLS.Z]))
    ax.xy_dataLim.update_from_data_xy(xy_bounds)

    z_bounds = np.vstack(((min_bounding_box[COLS.Z], min_bounding_box[COLS.Z]),
                          (max_bounding_box[COLS.Z], max_bounding_box[COLS.Z])))
    ax.zz_dataLim.update_from_data_xy(z_bounds)


def soma3d(ax, sm, color=None, alpha=_ALPHA, **_):
    '''Generates a 3d figure of the soma.

    Parameters:
        soma: Soma neurom.Soma object
    '''
    color = common.get_color(color, tree_type=NeuriteType.soma)

    if isinstance(sm, SomaCylinders):
        for start, end in zip(sm.points, sm.points[1:]):
            common.plot_cylinder(ax,
                                 start=start[COLS.XYZ], end=end[COLS.XYZ],
                                 start_radius=start[COLS.R], end_radius=end[COLS.R],
                                 color=color, alpha=alpha)
    else:
        common.plot_sphere(ax, center=sm.center[COLS.XYZ], radius=sm.radius,
                           color=color, alpha=alpha)


def neuron3d(ax, nrn,
             diameter_scale=_DIAMETER_SCALE, linewidth=_LINEWIDTH,
             color=None, alpha=_ALPHA,
             **_):
    '''
    Generates a figure of the neuron,
    that contains a soma and a list of trees.

    Parameters:
        neuron: Neuron \
        neurom.Neuron object
    '''
    soma3d(ax, nrn.soma, color=color, alpha=alpha)

    for neurite in nrn.neurites:
        tree3d(ax, neurite,
               diameter_scale=diameter_scale, linewidth=linewidth,
               color=color, alpha=alpha)

    ax.set_title(nrn.name)


def _generate_collection(group, ax, ctype, colors):
    '''Render rectangle collection'''
    color = common.TREE_COLOR[ctype]

    # generate segment collection
    collection = PolyCollection(group, closed=False, antialiaseds=True,
                                edgecolors='face', facecolors=color)

    # add it to the axes
    ax.add_collection(collection)

    # dummy plot for the legend
    if color not in colors:
        label = str(ctype).replace('NeuriteType.', '').replace('_', ' ').capitalize()
        ax.plot((0., 0.), (0., 0.), c=color, label=label)
        colors.add(color)


def _render_dendrogram(dnd, ax, displacement):
    '''Renders dendrogram'''
    # set of unique colors that reflect the set of types of the neurites
    colors = set()

    for n, (indices, ctype) in enumerate(zip(dnd.groups, dnd.types)):

        # slice rectangles array for the current neurite
        group = dnd.data[indices[0]:indices[1]]

        if n > 0:
            # displace the neurites by half of their maximum x dimension
            # plus half of the previous neurite's maxmimum x dimension
            displacement += 0.5 * (dnd.dims[n - 1][0] + dnd.dims[n][0])

        # arrange the trees without overlapping with each other
        group += (displacement, 0.)

        # create the polygonal collection of the dendrogram
        # segments
        _generate_collection(group, ax, ctype, colors)

    soma_square = dnd.soma

    if soma_square is not None:

        _generate_collection((soma_square + (displacement / 2., 0.),), ax, NeuriteType.soma, colors)
        ax.plot((displacement / 2., displacement), (0., 0.), color='k')
        ax.plot((0., displacement / 2.), (0., 0.), color='k')

    return displacement


def dendrogram(obj, show_diameters=True, new_fig=True, new_axes=True, subplot=False, **kwargs):
    '''
    Generates a figure of the neuron,
    that contains a soma and a list of trees.

    Parameters:
        obj: Neuron or tree \
        neurom.Neuron, neurom.Tree
        show_diameters : boolean \
            Determines if node diameters will \
            be show or not.
        new_fig: boolean \
            Defines if the tree will be plotted \
            in the current figure (False) \
            or in a new figure (True) \
            Default value is True.
        subplot: matplotlib subplot value or False \
            If False the default subplot 111 will be used. \
            For any other value a matplotlib subplot \
            will be generated. \
            Default value is False.

    Returns:
        A 2D matplotlib figure with a dendrogram view.
    '''

    # create dendrogram and generate rectangle collection
    dnd = Dendrogram(obj, show_diameters=show_diameters)
    dnd.generate()

    fig, ax = common.get_figure(new_fig=new_fig, new_axes=new_axes, subplot=subplot)

    # render dendrogram and take into account neurite displacement which
    # starts as zero. It is important to avoid overlapping of neurites
    # and to determine tha limits of the figure.

    displacement = _render_dendrogram(dnd, ax, 0.)

    # customization settings
    kwargs['xlim'] = [- dnd.dims[0][0] * 0.5, dnd.dims[-1][0] * 0.5 + displacement]

    kwargs.setdefault('title', 'Morphology Dendrogram')
    kwargs.setdefault('xlabel', 'micrometers (um)')
    kwargs.setdefault('ylabel', 'micrometers (um)')
    kwargs['no_legend'] = False
    kwargs.setdefault('aspect_ratio', 'auto')

    common.plot_style(fig=fig, ax=ax, **kwargs)
    return fig, ax

neuron.__doc__ += DEFAULT_PARAMS  # pylint: disable=no-member
tree.__doc__ += DEFAULT_PARAMS  # pylint: disable=no-member
soma.__doc__ += DEFAULT_PARAMS  # pylint: disable=no-member
neuron3d.__doc__ += DEFAULT_PARAMS  # pylint: disable=no-member
tree3d.__doc__ += DEFAULT_PARAMS  # pylint: disable=no-member
soma3d.__doc__ += DEFAULT_PARAMS  # pylint: disable=no-member
