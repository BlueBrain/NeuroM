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


from neurom.view import common
from neurom.core.types import TreeType
from matplotlib.collections import LineCollection
import numpy as np
from neurom.core.tree import isegment
from neurom.core.tree import val_iter
from neurom.io.readers import COLS
from neurom.analysis.morphtree import get_bounding_box
from neurom.analysis.morphtree import get_tree_type
from neurom.analysis.morphtree import i_segment_radius


def get_default(variable, **kwargs):
    """
    Returns default variable or kwargs variable if it exists.
    """
    default = {'linewidth': 1.2,
               'alpha': 0.8,
               'treecolor': None,
               'diameter': True,
               'diameter_scale': 1.0,
               'white_space': 30.}

    return kwargs.get(variable, default[variable])


def tree(tr, plane='xy', new_fig=True, subplot=False, **kwargs):

    """
    Generates a 2d figure of the tree.

    Parameters
    ----------
    tr: Tree
        neurom.Tree object

    Options
    -------
    plane: str
        Accepted values: Any pair of of xyz
        Default value is 'xy'.treecolor

    linewidth: float
        Defines the linewidth of the tree,
        if diameter is set to False.
        Default value is 1.2.

    alpha: float
        Defines the transparency of the tree.
        0.0 transparent through 1.0 opaque.
        Default value is 0.8.

    treecolor: str or None
        Defines the color of the tree.
        If None the default values will be used,
        depending on the type of tree:
        Basal dendrite: "red"
        Axon : "blue"
        Apical dendrite: "purple"
        Undefined tree: "black"
        Default value is None.

    new_fig: boolean
        Defines if the tree will be plotted
        in the current figure (False)
        or in a new figure (True)
        Default value is True.

    subplot: matplotlib subplot value or False
        If False the default subplot 111 will be used.
        For any other value a matplotlib subplot
        will be generated.
        Default value is False.

    diameter: boolean
        If True the diameter, scaled with diameter_scale factor,
        will define the width of the tree lines.
        If False use linewidth to select the width of the tree lines.
        Default value is True.

    diameter_scale: float
        Defines the scale factor that will be multiplied
        with the diameter to define the width of the tree line.
        Default value is 1.

    limits: list or boolean
        List of type: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        If False the figure will not be scaled.
        If True the figure will be scaled according to tree limits.
        Default value is False.

    white_space: float
        Defines the white space around
        the boundary box of the morphology.
        Default value is 1.

    Returns
    --------
    A 2D matplotlib figure with a tree view, at the selected plane.

    """
    if plane not in ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'):
        return None, 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.'

    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    # Data needed for the viewer: x,y,z,r
    bounding_box = get_bounding_box(tr)

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

    segs = [_seg_2d(seg) for seg in val_iter(isegment(tr))]

    linewidth = get_default('linewidth', **kwargs)
    # Definition of the linewidth according to diameter, if diameter is True.
    if get_default('diameter', **kwargs):
        scale = get_default('diameter_scale', **kwargs)
        # TODO: This was originally a numpy array. Did it have to be one?
        linewidth = [2 * d * scale for d in i_segment_radius(tr)]

    # Plot the collection of lines.
    collection = LineCollection(segs,
                                color=common.get_color(get_default('treecolor', **kwargs),
                                                       get_tree_type(tr)),
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

    """
    Generates a 2d figure of the soma.

    Parameters
    ----------
    soma: Soma
        neurom.Soma object

    Options
    -------
    plane: str
        Accepted values: Any pair of of xyz
        Default value is 'xy'

    linewidth: float
        Defines the linewidth of the soma.
        Default value is 1.2

    alpha: float
        Defines the transparency of the soma.
        0.0 transparent through 1.0 opaque.
        Default value is 0.8.

    treecolor: str or None
        Defines the color of the soma.
        If None the default value will be used:
        Soma : "black".
        Default value is None.

    new_fig: boolean
        Defines if the tree will be plotted
        in the current figure (False)
        or in a new figure (True)
        Default value is True.

    subplot: matplotlib subplot value or False
        If False the default subplot 111 will be used.
        For any other value a matplotlib subplot
        will be generated.
        Default value is False.

    limits: list or boolean
        List of type: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        If False the figure will not be scaled.
        If True the figure will be scaled according to tree limits.
        Default value is False.

    Returns
    --------
    A 2D matplotlib figure with a soma view, at the selected plane.

    """

    treecolor = kwargs.get('treecolor', None)
    outline = kwargs.get('outline', True)

    if plane not in ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'):
        return None, 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.'

    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    # Definition of the tree color depending on the tree type.
    treecolor = common.get_color(treecolor, tree_type=TreeType.soma)

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

    """
    Generates a 2d figure of the neuron,
    that contains a soma and a list of trees.

    Parameters
    ----------
    neuron: Neuron
        neurom.Neuron object

    Options
    -------
    plane: str
        Accepted values: Any pair of of xyz
        Default value is 'xy'

    linewidth: float
        Defines the linewidth of the tree and soma
        of the neuron, if diameter is set to False.
        Default value is 1.2.

    alpha: float
        Defines the transparency of the neuron.
        0.0 transparent through 1.0 opaque.
        Default value is 0.8.

    treecolor: str or None
        Defines the color of the trees.
        If None the default values will be used,
        depending on the type of tree:
        Soma: "black"
        Basal dendrite: "red"
        Axon : "blue"
        Apical dendrite: "purple"
        Undefined tree: "black"
        Default value is None.

    new_fig: boolean
        Defines if the neuron will be plotted
        in the current figure (False)
        or in a new figure (True)
        Default value is True.

    subplot: matplotlib subplot value or False
        If False the default subplot 111 will be used.
        For any other value a matplotlib subplot
        will be generated.
        Default value is False.

    diameter: boolean
        If True the diameter, scaled with diameter_scale factor,
        will define the width of the tree lines.
        If False use linewidth to select the width of the tree lines.
        Default value is True.

    diameter_scale: float
        Defines the scale factor that will be multiplied
        with the diameter to define the width of the tree line.
        Default value is 1.

    limits: list or boolean
        List of type: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        If False the figure will not be scaled.
        If True the figure will be scaled according to tree limits.
        Default value is False.

    Returns
    --------
    A 3D matplotlib figure with a tree view, at the selected plane.

    """
    if plane not in ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'):
        return None, 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.'

    new_fig = kwargs.get('new_fig', True)
    subplot = kwargs.get('subplot', False)

    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    kwargs['new_fig'] = False

    soma(nrn.soma, plane=plane, **kwargs)

    h = []
    v = []

    for temp_tree in nrn.neurites:

        bounding_box = get_bounding_box(temp_tree)

        h.append([bounding_box[0][getattr(COLS, plane[0].capitalize())],
                  bounding_box[1][getattr(COLS, plane[0].capitalize())]])
        v.append([bounding_box[0][getattr(COLS, plane[1].capitalize())],
                  bounding_box[1][getattr(COLS, plane[1].capitalize())]])

        tree(temp_tree, plane=plane, **kwargs)

    kwargs['title'] = kwargs.get('title', 'Neuron view')
    kwargs['xlabel'] = kwargs.get('xlabel', plane[0])
    kwargs['ylabel'] = kwargs.get('ylabel', plane[1])
    kwargs['xlim'] = kwargs.get('xlim', [np.min(h) - get_default('white_space', **kwargs),
                                         np.max(h) + get_default('white_space', **kwargs)])
    kwargs['ylim'] = kwargs.get('ylim', [np.min(v) - get_default('white_space', **kwargs),
                                         np.max(v) + get_default('white_space', **kwargs)])

    return common.plot_style(fig=fig, ax=ax, **kwargs)


def tree3d(tr, new_fig=True, new_axes=True, subplot=False, **kwargs):

    """
    Generates a figure of the tree in 3d.

    Parameters
    ----------
    tr: Tree
        neurom.Tree object

    Options
    -------
    linewidth: float
        Defines the linewidth of the tree,
        if diameter is set to False.
        Default value is 1.2.

    alpha: float
        Defines the transparency of the tree.
        0.0 transparent through 1.0 opaque.
        Default value is 0.8.

    treecolor: str or None
        Defines the color of the tree.
        If None the default values will be used,
        depending on the type of tree:
        Basal dendrite: "red"
        Axon : "blue"
        Apical dendrite: "purple"
        Undefined tree: "black"
        Default value is None.

    new_fig: boolean
        Defines if the tree will be plotted
        in the current figure (False)
        or in a new figure (True)
        Default value is True.

    subplot: matplotlib subplot value or False
        If False the default subplot 111 will be used.
        For any other value a matplotlib subplot
        will be generated.
        Default value is False.

    diameter: boolean
        If True the diameter, scaled with diameter_scale factor,
        will define the width of the tree lines.
        If False use linewidth to select the width of the tree lines.
        Default value is True.

    diameter_scale: float
        Defines the scale factor that will be multiplied
        with the diameter to define the width of the tree line.
        Default value is 1.

    white_space: float
        Defines the white space around
        the boundary box of the morphology.
        Default value is 1.

    Returns
    --------
    A 3D matplotlib figure with a tree view.

    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, new_axes=new_axes,
                                subplot=subplot, params={'projection': '3d'})

    # Data needed for the viewer: x,y,z,r
    bounding_box = get_bounding_box(tr)

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

    segs = [_seg_3d(seg) for seg in val_iter(isegment(tr))]

    linewidth = get_default('linewidth', **kwargs)

    # Definition of the linewidth according to diameter, if diameter is True.
    if get_default('diameter', **kwargs):
        # TODO: This was originally a numpy array. Did it have to be one?
        linewidth = [2 * d * get_default('diameter_scale', **kwargs) for d in i_segment_radius(tr)]

    # Plot the collection of lines.
    collection = Line3DCollection(segs,
                                  color=common.get_color(get_default('treecolor', **kwargs),
                                                         get_tree_type(tr)),
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

    """
    Generates a 3d figure of the soma.

    Parameters
    ----------
    soma: Soma
        neurom.Soma object

    Options
    -------
    alpha: float
        Defines the transparency of the soma.
        0.0 transparent through 1.0 opaque.
        Default value is 0.8.

    treecolor: str or None
        Defines the color of the soma.
        If None the default value will be used:
        Soma : "black".
        Default value is None.

    new_fig: boolean
        Defines if the tree will be plotted
        in the current figure (False)
        or in a new figure (True)
        Default value is True.

    subplot: matplotlib subplot value or False
        If False the default subplot 111 will be used.
        For any other value a matplotlib subplot
        will be generated.
        Default value is False.

    Returns
    --------
    A 3D matplotlib figure with a soma view.

    """

    treecolor = kwargs.get('treecolor', None)

    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, new_axes=new_axes,
                                subplot=subplot, params={'projection': '3d'})

    # Definition of the tree color depending on the tree type.
    treecolor = common.get_color(treecolor, tree_type=TreeType.soma)

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

    """
    Generates a figure of the neuron,
    that contains a soma and a list of trees.

    Parameters
    ----------
    neuron: Neuron
        neurom.Neuron object

    Options
    -------
    linewidth: float
        Defines the linewidth of the tree and soma
        of the neuron, if diameter is set to False.
        Default value is 1.2.

    alpha: float
        Defines the transparency of the neuron.
        0.0 transparent through 1.0 opaque.
        Default value is 0.8.

    treecolor: str or None
        Defines the color of the trees.
        If None the default values will be used,
        depending on the type of tree:
        Soma: "black"
        Basal dendrite: "red"
        Axon : "blue"
        Apical dendrite: "purple"
        Undefined tree: "black"
        Default value is None.

    new_fig: boolean
        Defines if the neuron will be plotted
        in the current figure (False)
        or in a new figure (True)
        Default value is True.

    subplot: matplotlib subplot value or False
        If False the default subplot 111 will be used.
        For any other value a matplotlib subplot
        will be generated.
        Default value is False.

    diameter: boolean
        If True the diameter, scaled with diameter_scale factor,
        will define the width of the tree lines.
        If False use linewidth to select the width of the tree lines.
        Default value is True.

    diameter_scale: float
        Defines the scale factor that will be multiplied
        with the diameter to define the width of the tree line.
        Default value is 1.

    Returns
    --------
    A 3D matplotlib figure with a neuron view.

    """
    new_fig = kwargs.get('new_fig', True)
    subplot = kwargs.get('subplot', False)

    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, new_axes=new_axes,
                                subplot=subplot, params={'projection': '3d'})

    kwargs['new_fig'] = False
    kwargs['new_axes'] = False

    soma3d(nrn.soma, **kwargs)

    h = []
    v = []
    d = []

    for temp_tree in nrn.neurites:

        bounding_box = get_bounding_box(temp_tree)

        h.append([bounding_box[0][getattr(COLS, 'X')],
                  bounding_box[1][getattr(COLS, 'X')]])
        v.append([bounding_box[0][getattr(COLS, 'Y')],
                  bounding_box[1][getattr(COLS, 'Y')]])
        d.append([bounding_box[0][getattr(COLS, 'Z')],
                  bounding_box[1][getattr(COLS, 'Z')]])

        tree3d(temp_tree, **kwargs)

    kwargs['title'] = kwargs.get('title', 'Neuron view')

    kwargs['xlim'] = kwargs.get('xlim', [np.min(h) - get_default('white_space', **kwargs),
                                         np.max(h) + get_default('white_space', **kwargs)])
    kwargs['ylim'] = kwargs.get('ylim', [np.min(v) - get_default('white_space', **kwargs),
                                         np.max(v) + get_default('white_space', **kwargs)])
    kwargs['zlim'] = kwargs.get('zlim', [np.min(d) - get_default('white_space', **kwargs),
                                         np.max(d) + get_default('white_space', **kwargs)])

    return common.plot_style(fig=fig, ax=ax, **kwargs)
