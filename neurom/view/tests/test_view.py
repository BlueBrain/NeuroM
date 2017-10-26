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
from .utils import get_fig_2d, get_fig_3d  # needs to be at top to trigger matplotlib Agg backend
import os

import itertools as it
import numpy as np

from nose import tools as nt
from neurom import load_neuron
from neurom.view import view, common
from neurom.core import Section
from neurom.core._soma import make_soma, SOMA_CONTOUR, SOMA_CYLINDER
from neurom.core.types import NeuriteType


DATA_PATH = './test_data'
SWC_PATH = os.path.join(DATA_PATH, 'swc/')
fst_neuron = load_neuron(os.path.join(SWC_PATH, 'Neuron.swc'))
simple_neuron = load_neuron(os.path.join(SWC_PATH, 'simple.swc'))


def test_tree():
    with get_fig_2d() as (fig, ax):
        tree = fst_neuron.neurites[0]
        view.plot_tree(ax, tree, color='black', diameter_scale=None, alpha=1., linewidth=1.2)
        collection = ax.collections[0]
        nt.eq_(collection.get_linewidth()[0], 1.2)
        np.testing.assert_allclose(collection.get_color(), np.array([[0., 0., 0., 1.]]))

    with get_fig_2d() as (fig, ax):
        nt.assert_raises(AssertionError, view.plot_tree, ax, tree, plane='wrong')

    with get_fig_2d() as (fig, ax):
        tree = simple_neuron.neurites[0]
        view.plot_tree(ax, tree)
        np.testing.assert_allclose(ax.dataLim.bounds, (-5., 0., 11., 5.), atol=1e-10)


def test_neuron():
    with get_fig_2d() as (fig, ax):
        view.plot_neuron(ax, fst_neuron)
        nt.ok_(ax.get_title() == fst_neuron.name)
        np.testing.assert_allclose(ax.dataLim.get_points(),
                                   [[-40.32853516, -57.600172],
                                    [64.74726272, 48.51626225], ])

    with get_fig_2d() as (fig, ax):
        nt.assert_raises(AssertionError, view.plot_tree, ax, fst_neuron, plane='wrong')


def test_tree3d():
    with get_fig_3d() as (fig, ax):
        tree = simple_neuron.neurites[0]
        view.plot_tree3d(ax, tree)
        xy_bounds = ax.xy_dataLim.bounds
        np.testing.assert_allclose(xy_bounds, (-5., 0., 11., 5.))
        zz_bounds = ax.zz_dataLim.bounds
        np.testing.assert_allclose(zz_bounds, (0., 0., 1., 1.))


def test_neuron3d():
    with get_fig_3d() as (fig, ax):
        view.plot_neuron3d(ax, fst_neuron)
        nt.ok_(ax.get_title() == fst_neuron.name)
        np.testing.assert_allclose(ax.xy_dataLim.get_points(),
                                   [[-40.32853516, -57.600172],
                                    [64.74726272, 48.51626225], ])
        np.testing.assert_allclose(ax.zz_dataLim.get_points().T[0],
                                   (-00.09999862, 54.20408797))


def test_neuron_no_neurites():
    filename = os.path.join(SWC_PATH, 'point_soma.swc')
    with get_fig_2d() as (fig, ax):
        view.plot_neuron(ax, load_neuron(filename))


def test_neuron3d_no_neurites():
    filename = os.path.join(SWC_PATH, 'point_soma.swc')
    with get_fig_3d() as (fig, ax):
        view.plot_neuron3d(ax, load_neuron(filename))


def test_dendrogram():
    with get_fig_2d() as (fig, ax):
        view.plot_dendrogram(ax, fst_neuron)
        np.testing.assert_allclose(ax.get_xlim(), (-20., 100.), rtol=0.25)


def test_one_point_branch():
    test_section = Section(points=np.array([[1., 1., 1., 0.5, 2, 1, 0]]))
    for diameter_scale, linewidth in it.product((1.0, None),
                                                (0.0, 1.2)):
        with get_fig_2d() as (fig, ax):
            view.plot_tree(ax, test_section, diameter_scale=diameter_scale, linewidth=linewidth)
        with get_fig_3d() as (fig, ax):
            view.plot_tree3d(ax, test_section, diameter_scale=diameter_scale, linewidth=linewidth)


soma0 = fst_neuron.soma

# upright, varying radius
soma_2pt_normal_pts = np.array([
    [0.0,   0.0,  0.0, 1.0,  1, 1, -1],
    [0.0,  10.0,  0.0, 10.0, 1, 2,  1],
])
soma_2pt_normal = make_soma(soma_2pt_normal_pts, soma_class=SOMA_CYLINDER)

# upright, uniform radius, multiple cylinders
soma_3pt_normal_pts = np.array([
    [0.0, -10.0,  0.0, 10.0, 1, 1, -1],
    [0.0,   0.0,  0.0, 10.0, 1, 2,  1],
    [0.0,   10.0, 0.0, 10.0, 1, 3,  2],
])
soma_3pt_normal = make_soma(soma_3pt_normal_pts, soma_class=SOMA_CYLINDER)

# increasing radius, multiple cylinders
soma_4pt_normal_pts = np.array([
    [0.0,   0.0,     0.0, 1.0, 1, 1, -1],
    [0.0,   -10.0,   0.0, 2.0, 1, 2, 1],
    [0.0,   -10.0, -10.0, 3.0, 1, 3, 2],
    [-10.0, -10.0, -10.0, 4.0, 1, 4, 3],
])
soma_4pt_normal_cylinder = make_soma(soma_4pt_normal_pts, soma_class=SOMA_CYLINDER)
soma_4pt_normal_contour = make_soma(soma_4pt_normal_pts, soma_class=SOMA_CONTOUR)


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
        np.testing.assert_allclose(ax.get_xlim(), (-10.,  10.), atol=2)
        np.testing.assert_allclose(ax.get_ylim(), (0.,  10.), atol=2)
        np.testing.assert_allclose(ax.get_zlim(), (-10.,  10.), atol=2)


def test_get_color():
    nt.eq_(view._get_color(None, NeuriteType.basal_dendrite), "red")
    nt.eq_(view._get_color(None, NeuriteType.axon), "blue")
    nt.eq_(view._get_color(None, NeuriteType.apical_dendrite), "purple")
    nt.eq_(view._get_color(None, NeuriteType.soma), "black")
    nt.eq_(view._get_color(None, NeuriteType.undefined), "green")
    nt.eq_(view._get_color(None, 'wrong'), "green")
    nt.eq_(view._get_color('blue', 'wrong'), "blue")
    nt.eq_(view._get_color('yellow', NeuriteType.axon), "yellow")
