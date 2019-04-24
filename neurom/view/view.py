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
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 501ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''visualize morphologies'''

from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import \
    Line3DCollection  # pylint: disable=relative-import

import numpy as np
from neurom import NeuriteType, geom
from neurom._compat import zip
from neurom.core import iter_neurites, iter_segments
from neurom.core._soma import SomaCylinders
from neurom.core.dataformat import COLS
from neurom.core.types import tree_type_checker
from neurom.morphmath import segment_radius
from neurom.view._dendrogram import Dendrogram

from . import common

_LINEWIDTH = 1.2
_ALPHA = 0.8
_DIAMETER_SCALE = 1.0
TREE_COLOR = {NeuriteType.basal_dendrite: 'red',
              NeuriteType.apical_dendrite: 'purple',
              NeuriteType.axon: 'blue',
              NeuriteType.soma: 'black',
              NeuriteType.undefined: 'green'}


def _plane2col(plane):
    '''take a string like 'xy', and return the indices from COLS.*'''
    planes = ('xy', 'yx', 'xz', 'zx', 'yz', 'zy')
    assert plane in planes, 'No such plane found! Please select one of: ' + str(planes)
    return (getattr(COLS, plane[0].capitalize()),
            getattr(COLS, plane[1].capitalize()), )


def _get_linewidth(tree, linewidth, diameter_scale):
    '''calculate the desired linewidth based on tree contents

    If diameter_scale exists, it is used to scale the diameter of each of the segments
    in the tree
    If diameter_scale is None, the linewidth is used.
    '''
    if diameter_scale is not None and tree:
        linewidth = [2 * segment_radius(s) * diameter_scale
                     for s in iter_segments(tree)]
    return linewidth


def _get_color(treecolor, tree_type):
    """if treecolor set, it's returned, otherwise tree_type is used to return set colors"""
    if treecolor is not None:
        return treecolor
    return TREE_COLOR.get(tree_type, 'green')


def plot_tree(ax, tree, plane='xy',
              diameter_scale=_DIAMETER_SCALE, linewidth=_LINEWIDTH,
              color=None, alpha=_ALPHA):
    '''Plots a 2d figure of the tree's segments

    Args:
        ax(matplotlib axes): on what to plot
        tree(neurom.core.Tree or neurom.core.Neurite): plotted tree
        plane(str): Any pair of 'xyz'
        diameter_scale(float): Scale factor multiplied with segment diameters before plotting
        linewidth(float): all segments are plotted with this width, but only if diameter_scale=None
        color(str or None): Color of plotted values, None corresponds to default choice
        alpha(float): Transparency of plotted values

    Note:
        If the tree contains one single point the plot will be empty
        since no segments can be constructed.
    '''
    plane0, plane1 = _plane2col(plane)
    segs = [((s[0][plane0], s[0][plane1]),
             (s[1][plane0], s[1][plane1]))
            for s in iter_segments(tree)]

    linewidth = _get_linewidth(tree, diameter_scale=diameter_scale, linewidth=linewidth)
    color = _get_color(color, tree.type)

    collection = LineCollection(segs, color=color, linewidth=linewidth, alpha=alpha)
    ax.add_collection(collection)


def plot_soma(ax, soma, plane='xy',
              soma_outline=True,
              linewidth=_LINEWIDTH,
              color=None, alpha=_ALPHA):
    '''Generates a 2d figure of the soma.

    Args:
        ax(matplotlib axes): on what to plot
        soma(neurom.core.Soma): plotted soma
        plane(str): Any pair of 'xyz'
        diameter_scale(float): Scale factor multiplied with segment diameters before plotting
        linewidth(float): all segments are plotted with this width, but only if diameter_scale=None
        color(str or None): Color of plotted values, None corresponds to default choice
        alpha(float): Transparency of plotted values
    '''
    plane0, plane1 = _plane2col(plane)
    color = _get_color(color, tree_type=NeuriteType.soma)

    if isinstance(soma, SomaCylinders):
        plane0, plane1 = _plane2col(plane)
        for start, end in zip(soma.points, soma.points[1:]):
            common.project_cylinder_onto_2d(ax, (plane0, plane1),
                                            start=start[COLS.XYZ], end=end[COLS.XYZ],
                                            start_radius=start[COLS.R], end_radius=end[COLS.R],
                                            color=color, alpha=alpha)
    else:
        if soma_outline:
            ax.add_artist(Circle(soma.center[[plane0, plane1]], soma.radius,
                                 color=color, alpha=alpha))
        else:
            plane0, plane1 = _plane2col(plane)
            points = [(p[plane0], p[plane1]) for p in soma.iter()]

            if points:
                points.append(points[0])  # close the loop
                ax.plot(points, color=color, alpha=alpha, linewidth=linewidth)

    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])

    bounding_box = geom.bounding_box(soma)
    ax.dataLim.update_from_data_xy(np.vstack(([bounding_box[0][plane0], bounding_box[0][plane1]],
                                              [bounding_box[1][plane0], bounding_box[1][plane1]])),
                                   ignore=False)


# pylint: disable=too-many-arguments
def plot_neuron(ax, nrn,
                neurite_type=NeuriteType.all,
                plane='xy',
                soma_outline=True,
                diameter_scale=_DIAMETER_SCALE, linewidth=_LINEWIDTH,
                color=None, alpha=_ALPHA):
    '''Plots a 2D figure of the neuron, that contains a soma and the neurites

    Args:
        ax(matplotlib axes): on what to plot
        neurite_type(NeuriteType): an optional filter on the neurite type
        nrn(neuron): neuron to be plotted
        soma_outline(bool): should the soma be drawn as an outline
        plane(str): Any pair of 'xyz'
        diameter_scale(float): Scale factor multiplied with segment diameters before plotting
        linewidth(float): all segments are plotted with this width, but only if diameter_scale=None
        color(str or None): Color of plotted values, None corresponds to default choice
        alpha(float): Transparency of plotted values
    '''
    plot_soma(ax, nrn.soma, plane=plane, soma_outline=soma_outline, linewidth=linewidth,
              color=color, alpha=alpha)

    for neurite in iter_neurites(nrn, filt=tree_type_checker(neurite_type)):
        plot_tree(ax, neurite, plane=plane,
                  diameter_scale=diameter_scale, linewidth=linewidth,
                  color=color, alpha=alpha)

    ax.set_title(nrn.name)
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])


def _update_3d_datalim(ax, obj):
    '''unlike w/ 2d Axes, the dataLim isn't set by collections, so it has to be updated manually'''
    min_bounding_box, max_bounding_box = geom.bounding_box(obj)
    xy_bounds = np.vstack((min_bounding_box[:COLS.Z],
                           max_bounding_box[:COLS.Z]))
    ax.xy_dataLim.update_from_data_xy(xy_bounds, ignore=False)

    z_bounds = np.vstack(((min_bounding_box[COLS.Z], min_bounding_box[COLS.Z]),
                          (max_bounding_box[COLS.Z], max_bounding_box[COLS.Z])))
    ax.zz_dataLim.update_from_data_xy(z_bounds, ignore=False)


def plot_tree3d(ax, tree,
                diameter_scale=_DIAMETER_SCALE, linewidth=_LINEWIDTH,
                color=None, alpha=_ALPHA):
    '''Generates a figure of the tree in 3d.

    If the tree contains one single point the plot will be empty \
    since no segments can be constructed.

    Args:
        ax(matplotlib axes): on what to plot
        tree(neurom.core.Tree or neurom.core.Neurite): plotted tree
        diameter_scale(float): Scale factor multiplied with segment diameters before plotting
        linewidth(float): all segments are plotted with this width, but only if diameter_scale=None
        color(str or None): Color of plotted values, None corresponds to default choice
        alpha(float): Transparency of plotted values
    '''
    segs = [(s[0][COLS.XYZ], s[1][COLS.XYZ]) for s in iter_segments(tree)]

    linewidth = _get_linewidth(tree, diameter_scale=diameter_scale, linewidth=linewidth)
    color = _get_color(color, tree.type)

    collection = Line3DCollection(segs, color=color, linewidth=linewidth, alpha=alpha)
    ax.add_collection3d(collection)

    _update_3d_datalim(ax, tree)


def plot_soma3d(ax, soma, color=None, alpha=_ALPHA):
    '''Generates a 3d figure of the soma.

    Args:
        ax(matplotlib axes): on what to plot
        soma(neurom.core.Soma): plotted soma
        color(str or None): Color of plotted values, None corresponds to default choice
        alpha(float): Transparency of plotted values
    '''
    color = _get_color(color, tree_type=NeuriteType.soma)

    if isinstance(soma, SomaCylinders):
        for start, end in zip(soma.points, soma.points[1:]):
            common.plot_cylinder(ax,
                                 start=start[COLS.XYZ], end=end[COLS.XYZ],
                                 start_radius=start[COLS.R], end_radius=end[COLS.R],
                                 color=color, alpha=alpha)
    else:
        common.plot_sphere(ax, center=soma.center[COLS.XYZ], radius=soma.radius,
                           color=color, alpha=alpha)

    # unlike w/ 2d Axes, the dataLim isn't set by collections, so it has to be updated manually
    _update_3d_datalim(ax, soma)


def plot_neuron3d(ax, nrn, neurite_type=NeuriteType.all,
                  diameter_scale=_DIAMETER_SCALE, linewidth=_LINEWIDTH,
                  color=None, alpha=_ALPHA):
    '''
    Generates a figure of the neuron,
    that contains a soma and a list of trees.

    Args:
        ax(matplotlib axes): on what to plot
        nrn(neuron): neuron to be plotted
        neurite_type(NeuriteType): an optional filter on the neurite type
        diameter_scale(float): Scale factor multiplied with segment diameters before plotting
        linewidth(float): all segments are plotted with this width, but only if diameter_scale=None
        color(str or None): Color of plotted values, None corresponds to default choice
        alpha(float): Transparency of plotted values
    '''
    plot_soma3d(ax, nrn.soma, color=color, alpha=alpha)

    for neurite in iter_neurites(nrn, filt=tree_type_checker(neurite_type)):
        plot_tree3d(ax, neurite,
                    diameter_scale=diameter_scale, linewidth=linewidth,
                    color=color, alpha=alpha)

    ax.set_title(nrn.name)


def _generate_collection(group, ax, ctype, colors):
    '''Render rectangle collection'''
    color = TREE_COLOR[ctype]

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


def plot_dendrogram(ax, obj, show_diameters=True):
    '''Dendrogram of `obj`

    Args:
        obj: Neuron or tree \
        neurom.Neuron, neurom.Tree
        show_diameters : boolean \
            Determines if node diameters will \
            be show or not.
    '''
    # create dendrogram and generate rectangle collection
    dnd = Dendrogram(obj, show_diameters=show_diameters)
    dnd.generate()

    # render dendrogram and take into account neurite displacement which
    # starts as zero. It is important to avoid overlapping of neurites
    # and to determine tha limits of the figure.

    _render_dendrogram(dnd, ax, 0.)

    ax.set_title('Morphology Dendrogram')
    ax.set_xlabel('micrometers (um)')
    ax.set_ylabel('micrometers (um)')

    ax.set_aspect('auto')
    ax.legend()
