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
import itertools as it
from pathlib import Path
import warnings

import numpy as np
from nose.tools import assert_raises, eq_, ok_
from numpy.testing import assert_allclose, assert_array_almost_equal
from io import StringIO

from neurom import load_neuron
from neurom.core import Neurite, Section
from neurom.core._soma import SOMA_CONTOUR, SOMA_CYLINDER
from neurom.core.types import NeuriteType
from neurom.view import common, view

from .utils import (  # needs to be at top to trigger matplotlib Agg backend
    get_fig_2d, get_fig_3d)

DATA_PATH = './test_data'
SWC_PATH = Path(DATA_PATH, 'swc/')
fst_neuron = load_neuron(Path(SWC_PATH, 'Neuron.swc'))
simple_neuron = load_neuron(Path(SWC_PATH, 'simple.swc'))


def test_tree():
    neuron = load_neuron(Path(SWC_PATH, 'simple-different-section-types.swc'))
    expected_colors = {'black': np.array([[0., 0., 0., 1.] for _ in range(3)]),
                       None: [[1.      , 0.      , 0.      , 1.],
                              [1.      , 0.      , 0.      , 1.],
                              [0.501961, 0.      , 0.501961, 1.]]}
    for input_color, expected_colors in expected_colors.items():
        with get_fig_2d() as (fig, ax):
            tree = neuron.neurites[0]
            view.plot_tree(ax, tree,
                           color=input_color, diameter_scale=None, alpha=1., linewidth=1.2)
            collection = ax.collections[0]
            eq_(collection.get_linewidth()[0], 1.2)
            assert_array_almost_equal(collection.get_colors(), expected_colors)

        with get_fig_2d() as (fig, ax):
            tree = neuron.neurites[0]
            view.plot_tree(ax, tree,
                           color=input_color, alpha=1., linewidth=1.2, realistic_diameters=True)
            collection = ax.collections[0]
            eq_(collection.get_linewidth()[0], 1.0)
            assert_array_almost_equal(collection.get_facecolors(), expected_colors)


    with get_fig_2d() as (fig, ax):
        assert_raises(AssertionError, view.plot_tree, ax, tree, plane='wrong')

    with get_fig_2d() as (fig, ax):
        tree = simple_neuron.neurites[0]
        view.plot_tree(ax, tree)
        assert_allclose(ax.dataLim.bounds, (-5., 0., 11., 5.), atol=1e-10)


def test_neuron():
    with get_fig_2d() as (fig, ax):
        view.plot_neuron(ax, fst_neuron)
        ok_(ax.get_title() == fst_neuron.name)
        assert_allclose(ax.dataLim.get_points(),
                                   [[-40.32853516, -57.600172],
                                    [64.74726272, 48.51626225], ])

    with get_fig_2d() as (fig, ax):
        assert_raises(AssertionError, view.plot_tree, ax, fst_neuron, plane='wrong')


def test_tree3d():
    with get_fig_3d() as (fig, ax):
        tree = simple_neuron.neurites[0]
        view.plot_tree3d(ax, tree)
        xy_bounds = ax.xy_dataLim.bounds
        assert_allclose(xy_bounds, (-5., 0., 11., 5.))
        zz_bounds = ax.zz_dataLim.bounds
        assert_allclose(zz_bounds, (0., 0., 1., 1.))


def test_neuron3d():
    with get_fig_3d() as (fig, ax):
        view.plot_neuron3d(ax, fst_neuron)
        ok_(ax.get_title() == fst_neuron.name)
        assert_allclose(ax.xy_dataLim.get_points(),
                                   [[-40.32853516, -57.600172],
                                    [64.74726272, 48.51626225], ])
        assert_allclose(ax.zz_dataLim.get_points().T[0],
                                   (-00.09999862, 54.20408797))


def test_neuron_no_neurites():
    filename = Path(SWC_PATH, 'point_soma.swc')
    with get_fig_2d() as (fig, ax):
        view.plot_neuron(ax, load_neuron(filename))


def test_neuron3d_no_neurites():
    filename = Path(SWC_PATH, 'point_soma.swc')
    with get_fig_3d() as (fig, ax):
        view.plot_neuron3d(ax, load_neuron(filename))


def test_dendrogram():
    with get_fig_2d() as (fig, ax):
        view.plot_dendrogram(ax, fst_neuron)
        assert_allclose(ax.get_xlim(), (-10., 180.), rtol=0.25)

    with get_fig_2d() as (fig, ax):
        view.plot_dendrogram(ax, fst_neuron, show_diameters=False)
        assert_allclose(ax.get_xlim(), (-10., 180.), rtol=0.25)


def test_one_point_branch():
    test_section = Neurite(Section(points=np.array([[1., 1., 1., 0.5, 2, 1, 0]])))
    for diameter_scale, linewidth in it.product((1.0, None),
                                                (0.0, 1.2)):
        with get_fig_2d() as (fig, ax):
            view.plot_tree(ax, test_section, diameter_scale=diameter_scale, linewidth=linewidth)
        with get_fig_3d() as (fig, ax):
            view.plot_tree3d(ax, test_section, diameter_scale=diameter_scale, linewidth=linewidth)

with warnings.catch_warnings(record=True):
    soma0 = fst_neuron.soma

    # upright, varying radius
    soma_2pt_normal = load_neuron(StringIO(u"""1 1 0  0 0 1  -1
                                               2 1 0 10 0 10  1"""), reader='swc').soma

    # upright, uniform radius, multiple cylinders
    soma_3pt_normal = load_neuron(StringIO(u"""1 1 0 -10 0 10  -1
                                               2 1 0   0 0 10   1
                                               3 1 0  10 0 10   2"""), reader='swc').soma

    # increasing radius, multiple cylinders
    soma_4pt_normal_cylinder = load_neuron(StringIO(u"""1 1   0   0   0 1 -1
                                                       2 1   0 -10   0 2  1
                                                       3 1   0 -10  10 4  2
                                                       4 1 -10 -10 -10 4  3"""), reader='swc').soma

    soma_4pt_normal_contour = load_neuron(StringIO(u"""((CellBody)
                                                       (0     0   0 1)
                                                       (0   -10   0 2)
                                                       (0   -10  10 4)
                                                       (-10 -10 -10 4))"""), reader='asc').soma


def test_soma():
    for s in (soma0,
              soma_2pt_normal,
              soma_3pt_normal,
              soma_4pt_normal_cylinder,
              soma_4pt_normal_contour):
        with get_fig_2d() as (fig, ax):
            view.plot_soma(ax, s)
            common.plt.close(fig)

        with get_fig_2d() as (fig, ax):
            view.plot_soma(ax, s, soma_outline=False)
            common.plt.close(fig)


def test_soma3d():
    with get_fig_3d() as (_, ax):
        view.plot_soma3d(ax, soma_2pt_normal)
        assert_allclose(ax.get_xlim(), (-10.,  10.), atol=2)
        assert_allclose(ax.get_ylim(), (0.,  10.), atol=2)
        assert_allclose(ax.get_zlim(), (-10.,  10.), atol=2)


def test_get_color():
    eq_(view._get_color(None, NeuriteType.basal_dendrite), "red")
    eq_(view._get_color(None, NeuriteType.axon), "blue")
    eq_(view._get_color(None, NeuriteType.apical_dendrite), "purple")
    eq_(view._get_color(None, NeuriteType.soma), "black")
    eq_(view._get_color(None, NeuriteType.undefined), "green")
    eq_(view._get_color(None, 'wrong'), "green")
    eq_(view._get_color('blue', 'wrong'), "blue")
    eq_(view._get_color('yellow', NeuriteType.axon), "yellow")
