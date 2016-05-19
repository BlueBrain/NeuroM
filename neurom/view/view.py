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

from itertools import izip
from neurom.view import common
from neurom.core.types import NeuriteType
from matplotlib.collections import LineCollection
import numpy as np
from neurom.io import COLS
from neurom import _compat
from neurom.analysis.morphmath import segment_radius


def get_default(variable, **kwargs):
    '''
    Returns default variable or kwargs variable if it exists.
    '''
    default = {'linewidth': 1.2,
               'alpha': 0.8,
               'treecolor': None,
               'diameter': True,
               'diameter_scale': 1.0,
               'white_space': 30.}

    return kwargs.get(variable, default[variable])


def tree(tr, plane='xy', new_fig=True, subplot=False, **kwargs):
    '''
    Generates a 2d figure of the tree's segments. \
    If the tree contains one single point the plot will be empty \
    since no segments can be constructed.

    Parameters:
        tr: Tree \
            neurom.Tree object
        plane: str \
            Accepted values: Any pair of of xyz \
            Default value is 'xy'.treecolor
        linewidth: float \
            Defines the linewidth of the tree, \
            if diameter is set to False. \
            Default value is 1.2.
        alpha: float \
            Defines throughe transparency of the tree. \
            0.0 transparent through 1.0 opaque. \
            Default value is 0.8.
        treecolor: str or None \
            Defines the color of the tree. \
            If None the default values will be used, \
            depending on the type of tree: \
            Basal dendrite: "red" \
            Axon : "blue" \
            Apical dendrite: "purple" \
            Undefined tree: "black" \
            Default value is None.
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
        diameter: boolean
            If True the diameter, scaled with diameter_scale factor, \
            will define the width of the tree lines. \
            If False use linewidth to select the width of the tree lines. \
            Default value is True.
        diameter_scale: float \
            Defines the scale factor that will be multiplied \
            with the diameter to define the width of the tree line. \
            Default value is 1.
        limits: list or boolean \
            List of type: [[xmin, ymin, zmin], [xmax, ymax, zmax]] \
            If False the figure will not be scaled. \
            If True the figure will be scaled according to tree limits. \
            Default value is False.
        white_space: float \
            Defines the white space around \
            the boundary box of the morphology. \
            Default value is 1.

    Returns:
        A 2D matplotlib figure with a tree view, at the selected plane.
    '''
    if plane not in ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'):
        return None, 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.'

    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    # Data needed for the viewer: x,y,z,r
    bounding_box = _compat.bounding_box(tr)

    def _seg_2d(seg):
        '''2d coordinates needed for the plotting of a segment'''
        horz = getattr(COLS, plane[0].capitalize())
        vert = getattr(COLS, plane[1].capitalize())
        parent_point = seg[0]
        child_point = seg[1]
        horz1 = parent_point[horz]
        horz2 = child_point[horz]
        vert1 = parent_point[vert]
        vert2 = child_point[vert]
        return ((horz1, vert1), (horz2, vert2))

    segs = _compat.map_segments(tr, _seg_2d)

    linewidth = get_default('linewidth', **kwargs)
    # Definition of the linewidth according to diameter, if diameter is True.
    if get_default('diameter', **kwargs):
        scale = get_default('diameter_scale', **kwargs)
        # TODO: This was originally a numpy array. Did it have to be one?
        linewidth = [2 * r * scale for r in _compat.map_segments(tr, segment_radius)]
        if len(linewidth) == 0:
            linewidth = get_default('linewidth', **kwargs)

    # Plot the collection of lines.
    collection = LineCollection(segs,
                                color=common.get_color(get_default('treecolor', **kwargs),
                                                       _compat.neurite_type(tr)),
                                linewidth=linewidth, alpha=get_default('alpha', **kwargs))

    ax.add_collection(collection)

    kwargs['title'] = kwargs.get('title', 'Tree view')
    kwargs['xlabel'] = kwargs.get('xlabel', plane[0])
    kwargs['ylabel'] = kwargs.get('ylabel', plane[1])
    kwargs['xlim'] = kwargs.get('xlim', [bounding_box[0][getattr(COLS, plane[0].capitalize())] -
                                         get_default('white_space', **kwargs),
                                         bounding_box[1][getattr(COLS, plane[0].capitalize())] +
                                         get_default('white_space', **kwargs)])
    kwargs['ylim'] = kwargs.get('ylim', [bounding_box[0][getattr(COLS, plane[1].capitalize())] -
                                         get_default('white_space', **kwargs),
                                         bounding_box[1][getattr(COLS, plane[1].capitalize())] +
                                         get_default('white_space', **kwargs)])

    return common.plot_style(fig=fig, ax=ax, **kwargs)


def soma(sm, plane='xy', new_fig=True, subplot=False, **kwargs):
    '''
    Generates a 2d figure of the soma.

    Parameters:
        soma: Soma \
        neurom.Soma object
        plane: str \
            Accepted values: Any pair of of xyz \
            Default value is 'xy'.treecolor
        linewidth: float \
            Defines the linewidth of the tree, \
            if diameter is set to False. \
            Default value is 1.2.
        alpha: float \
            Defines throughe transparency of the tree. \
            0.0 transparent through 1.0 opaque. \
            Default value is 0.8.
        treecolor: str or None \
            Defines the color of the soma. \
            Soma: black" \
            Default value is None.
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
        limits: list or boolean \
            List of type: [[xmin, ymin, zmin], [xmax, ymax, zmax]] \
            If False the figure will not be scaled. \
            If True the figure will be scaled according to tree limits. \
            Default value is False.

    Returns:
        A 2D matplotlib figure with a soma view, at the selected plane.
    '''
    treecolor = kwargs.get('treecolor', None)
    outline = kwargs.get('outline', True)

    if plane not in ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'):
        return None, 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.'

    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    # Definition of the tree color depending on the tree type.
    treecolor = common.get_color(treecolor, tree_type=NeuriteType.soma)

    # Plot the outline of the soma as a circle, is outline is selected.
    if not outline:
        soma_circle = common.plt.Circle(sm.center, sm.radius, color=treecolor,
                                        alpha=get_default('alpha', **kwargs))
        ax.add_artist(soma_circle)
    else:
        horz = []
        vert = []

        for s_point in sm.iter():
            horz.append(s_point[getattr(COLS, plane[0].capitalize())])
            vert.append(s_point[getattr(COLS, plane[1].capitalize())])

        horz.append(horz[0]) # To close the loop for a soma viewer. This might be modified!
        vert.append(vert[0]) # To close the loop for a soma viewer. This might be modified!

        common.plt.plot(horz, vert, color=treecolor,
                        alpha=get_default('alpha', **kwargs),
                        linewidth=get_default('linewidth', **kwargs))

    kwargs['title'] = kwargs.get('title', 'Soma view')
    kwargs['xlabel'] = kwargs.get('xlabel', plane[0])
    kwargs['ylabel'] = kwargs.get('ylabel', plane[1])

    return common.plot_style(fig=fig, ax=ax, **kwargs)


def neuron(nrn, plane='xy', new_fig=True, subplot=False, **kwargs):
    '''
    Generates a 2d figure of the neuron, \
    that contains a soma and a list of trees.

    Parameters:
        neuron: Neuron \
        neurom.Neuron object
        plane: str \
            Accepted values: Any pair of of xyz \
            Default value is 'xy'.treecolor
        linewidth: float \
            Defines the linewidth of the tree, \
            if diameter is set to False. \
            Default value is 1.2.
        alpha: float \
            Defines throughe transparency of the tree. \
            0.0 transparent through 1.0 opaque. \
            Default value is 0.8.
        treecolor: str or None \
            Defines the color of the tree. \
            If None the default values will be used, \
            depending on the type of tree: \
            Basal dendrite: "red" \
            Axon : "blue" \
            Apical dendrite: "purple" \
            Undefined tree: "black" \
            Default value is None.
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
        diameter: boolean
            If True the diameter, scaled with diameter_scale factor, \
            will define the width of the tree lines. \
            If False use linewidth to select the width of the tree lines. \
            Default value is True.
        diameter_scale: float \
            Defines the scale factor that will be multiplied \
            with the diameter to define the width of the tree line. \
            Default value is 1.
        limits: list or boolean \
            List of type: [[xmin, ymin, zmin], [xmax, ymax, zmax]] \
            If False the figure will not be scaled. \
            If True the figure will be scaled according to tree limits. \
            Default value is False.

    Returns:
        A 3D matplotlib figure with a tree view, at the selected plane.
    '''
    if plane not in ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'):
        return None, 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.'

    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    kwargs['new_fig'] = False
    kwargs['subplot'] = subplot

    kwargs['final'] = False

    soma(nrn.soma, plane=plane, **kwargs)

    kwargs['title'] = kwargs.get('title', 'Neuron view')
    kwargs['xlabel'] = kwargs.get('xlabel', plane[0])
    kwargs['ylabel'] = kwargs.get('ylabel', plane[1])

    h = []
    v = []

    for temp_tree in nrn.neurites:

        bounding_box = _compat.bounding_box(temp_tree)

        h.append([bounding_box[0][getattr(COLS, plane[0].capitalize())],
                  bounding_box[1][getattr(COLS, plane[0].capitalize())]])
        v.append([bounding_box[0][getattr(COLS, plane[1].capitalize())],
                  bounding_box[1][getattr(COLS, plane[1].capitalize())]])

        tree(temp_tree, plane=plane, **kwargs)

    if h:
        kwargs['xlim'] = kwargs.get('xlim', [np.min(h) - get_default('white_space', **kwargs),
                                             np.max(h) + get_default('white_space', **kwargs)])
    if v:
        kwargs['ylim'] = kwargs.get('ylim', [np.min(v) - get_default('white_space', **kwargs),
                                             np.max(v) + get_default('white_space', **kwargs)])

    kwargs['final'] = True
    return common.plot_style(fig=fig, ax=ax, **kwargs)


def tree3d(tr, new_fig=True, new_axes=True, subplot=False, **kwargs):
    '''
    Generates a figure of the tree in 3d.
    If the tree contains one single point the plot will be empty \
    since no segments can be constructed.


    Parameters:
        tr: Tree \
        neurom.Tree object
        linewidth: float \
            Defines the linewidth of the tree, \
            if diameter is set to False. \
            Default value is 1.2.
        alpha: float \
            Defines throughe transparency of the tree. \
            0.0 transparent through 1.0 opaque. \
            Default value is 0.8.
        treecolor: str or None \
            Defines the color of the tree. \
            If None the default values will be used, \
            depending on the type of tree: \
            Basal dendrite: "red" \
            Axon : "blue" \
            Apical dendrite: "purple" \
            Undefined tree: "black" \
            Default value is None.
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
        diameter: boolean \
            If True the diameter, scaled with diameter_scale factor, \
            will define the width of the tree lines. \
            If False use linewidth to select the width of the tree lines. \
            Default value is True.
        diameter_scale: float \
            Defines the scale factor that will be multiplied \
            with the diameter to define the width of the tree line. \
            Default value is 1.
        white_space: float \
            Defines the white space around \
            the boundary box of the morphology. \
            Default value is 1.

    Returns:
        A 3D matplotlib figure with a tree view.
    '''
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, new_axes=new_axes,
                                subplot=subplot, params={'projection': '3d'})

    # Data needed for the viewer: x,y,z,r
    bounding_box = _compat.bounding_box(tr)

    def _seg_3d(seg):
        '''2d coordinates needed for the plotting of a segment'''
        horz = getattr(COLS, 'X')
        vert = getattr(COLS, 'Y')
        depth = getattr(COLS, 'Z')
        parent_point = seg[0]
        child_point = seg[1]
        horz1 = parent_point[horz]
        horz2 = child_point[horz]
        vert1 = parent_point[vert]
        vert2 = child_point[vert]
        depth1 = parent_point[depth]
        depth2 = child_point[depth]
        return ((horz1, vert1, depth1), (horz2, vert2, depth2))

    segs = _compat.map_segments(tr, _seg_3d)

    linewidth = get_default('linewidth', **kwargs)

    # Definition of the linewidth according to diameter, if diameter is True.
    if get_default('diameter', **kwargs):
        # TODO: This was originally a numpy array. Did it have to be one?
        scale = get_default('diameter_scale', **kwargs)
        linewidth = [2 * r * scale for r in _compat.map_segments(tr, segment_radius)]
        if len(linewidth) == 0:
            linewidth = get_default('linewidth', **kwargs)

    # Plot the collection of lines.
    collection = Line3DCollection(segs,
                                  color=common.get_color(get_default('treecolor', **kwargs),
                                                         _compat.neurite_type(tr)),
                                  linewidth=linewidth, alpha=get_default('alpha', **kwargs))

    ax.add_collection3d(collection)

    kwargs['title'] = kwargs.get('title', 'Tree 3d-view')
    kwargs['xlabel'] = kwargs.get('xlabel', 'X')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Y')
    kwargs['zlabel'] = kwargs.get('zlabel', 'Z')
    kwargs['xlim'] = kwargs.get('xlim', [bounding_box[0][0] - get_default('white_space', **kwargs),
                                         bounding_box[1][0] + get_default('white_space', **kwargs)])
    kwargs['ylim'] = kwargs.get('ylim', [bounding_box[0][1] - get_default('white_space', **kwargs),
                                         bounding_box[1][1] + get_default('white_space', **kwargs)])
    kwargs['zlim'] = kwargs.get('zlim', [bounding_box[0][2] - get_default('white_space', **kwargs),
                                         bounding_box[1][2] + get_default('white_space', **kwargs)])

    return common.plot_style(fig=fig, ax=ax, **kwargs)


def soma3d(sm, new_fig=True, new_axes=True, subplot=False, **kwargs):
    '''
    Generates a 3d figure of the soma.

    Parameters:
        soma: Soma \
        neurom.Soma object
        alpha: float \
            Defines throughe transparency of the tree. \
            0.0 transparent through 1.0 opaque. \
            Default value is 0.8.
        treecolor: str or None \
            Defines the color of the soma. \
            Soma : "black". \
            Default value is None.
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
        A 3D matplotlib figure with a soma view.
    '''
    treecolor = kwargs.get('treecolor', None)

    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, new_axes=new_axes,
                                subplot=subplot, params={'projection': '3d'})

    # Definition of the tree color depending on the tree type.
    treecolor = common.get_color(treecolor, tree_type=NeuriteType.soma)

    xs = sm.center[0]
    ys = sm.center[1]
    zs = sm.center[2]

    # Plot the soma as a circle.
    fig, ax = common.plot_sphere(fig, ax, center=[xs, ys, zs], radius=sm.radius, color=treecolor,
                                 alpha=get_default('alpha', **kwargs))

    kwargs['title'] = kwargs.get('title', 'Soma view')
    kwargs['xlabel'] = kwargs.get('xlabel', 'X')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Y')
    kwargs['zlabel'] = kwargs.get('zlabel', 'Z')

    return common.plot_style(fig=fig, ax=ax, **kwargs)


def neuron3d(nrn, new_fig=True, new_axes=True, subplot=False, **kwargs):
    '''
    Generates a figure of the neuron,
    that contains a soma and a list of trees.

    Parameters:
        neuron: Neuron \
        neurom.Neuron object
        linewidth: float \
            Defines the linewidth of the tree, \
            if diameter is set to False. \
            Default value is 1.2.
        alpha: float \
            Defines throughe transparency of the tree. \
            0.0 transparent through 1.0 opaque. \
            Default value is 0.8.
        treecolor: str or None \
            Defines the color of the tree. \
            If None the default values will be used, \
            depending on the type of tree: \
            Basal dendrite: "red" \
            Axon : "blue" \
            Apical dendrite: "purple" \
            Undefined tree: "black" \
            Default value is None.
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
        diameter: boolean
            If True the diameter, scaled with diameter_scale factor, \
            will define the width of the tree lines. \
            If False use linewidth to select the width of the tree lines. \
            Default value is True.
        diameter_scale: float \
            Defines the scale factor that will be multiplied \
            with the diameter to define the width of the tree line. \
            Default value is 1.

    Returns:
        A 3D matplotlib figure with a neuron view.
    '''
    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, new_axes=new_axes,
                                subplot=subplot, params={'projection': '3d'})

    kwargs['new_fig'] = False
    kwargs['subplot'] = subplot
    kwargs['new_axes'] = False
    kwargs['title'] = kwargs.get('title', 'Neuron view')

    kwargs['final'] = False

    soma3d(nrn.soma, **kwargs)

    h = []
    v = []
    d = []

    for temp_tree in nrn.neurites:

        bounding_box = _compat.bounding_box(temp_tree)

        h.append([bounding_box[0][getattr(COLS, 'X')],
                  bounding_box[1][getattr(COLS, 'X')]])
        v.append([bounding_box[0][getattr(COLS, 'Y')],
                  bounding_box[1][getattr(COLS, 'Y')]])
        d.append([bounding_box[0][getattr(COLS, 'Z')],
                  bounding_box[1][getattr(COLS, 'Z')]])

        tree3d(temp_tree, **kwargs)

    if h:
        kwargs['xlim'] = kwargs.get('xlim', [np.min(h) - get_default('white_space', **kwargs),
                                             np.max(h) + get_default('white_space', **kwargs)])
    if v:
        kwargs['ylim'] = kwargs.get('ylim', [np.min(v) - get_default('white_space', **kwargs),
                                             np.max(v) + get_default('white_space', **kwargs)])
    if d:
        kwargs['zlim'] = kwargs.get('zlim', [np.min(d) - get_default('white_space', **kwargs),
                                             np.max(d) + get_default('white_space', **kwargs)])

    kwargs['final'] = True
    return common.plot_style(fig=fig, ax=ax, **kwargs)


def _format_str(string):
    ''' String formatting
    '''
    return string.replace('NeuriteType.', '').replace('_', ' ').capitalize()


def _generate_collection(group, ax, ctype, colors):
    ''' Render rectangle collection
    '''
    from matplotlib.collections import PolyCollection

    color = common.TREE_COLOR[ctype]

    # generate segment collection
    collection = PolyCollection(group, closed=False, antialiaseds=True,
                                edgecolors='face', facecolors=color)

    # add it to the axes
    ax.add_collection(collection)

    # dummy plot for the legend
    if color not in colors:
        ax.plot((0., 0.), (0., 0.), c=color, label=_format_str(str(ctype)))
        colors.add(color)


def _render_dendrogram(dnd, ax, displacement):
    '''Renders dendrogram
    '''
    # set of unique colors that reflect the set of types of the neurites
    colors = set()

    for n, (indices, ctype) in enumerate(izip(dnd.groups, dnd.types)):

        # slice rectangles array for the current neurite
        group = dnd.data[indices[0]: indices[1]]

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
        linewidth: float \
            Defines the linewidth of the tree, \
            if diameter is set to False. \
            Default value is 1.2.
        alpha: float \
            Defines throughe transparency of the tree. \
            0.0 transparent through 1.0 opaque. \
            Default value is 0.8.
        treecolor: str or None \
            Defines the color of the tree. \
            If None the default values will be used, \
            depending on the type of tree: \
            Basal dendrite: "red" \
            Axon : "blue" \
            Apical dendrite: "purple" \
            Undefined tree: "black" \
            Default value is None.
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
    from neurom.analysis.dendrogram import Dendrogram

    if _compat.is_new_style(obj):
        raise NotImplementedError('dendrogram not implemented for fst.Neuron')

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

    kwargs['title'] = kwargs.get('title', 'Morphology Dendrogram')
    kwargs['xlabel'] = kwargs.get('xlabel', 'micrometers (um)')
    kwargs['ylabel'] = kwargs.get('ylabel', 'micrometers (um)')
    kwargs['no_legend'] = False

    return common.plot_style(fig=fig, ax=ax, **kwargs)
