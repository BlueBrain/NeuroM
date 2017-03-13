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
import os
import itertools as it
import numpy as np

from nose import tools as nt
from neurom import load_neuron
from neurom.view import view, common
from neurom.core import Section
from neurom.core._soma import make_soma, SOMA_CONTOUR, SOMA_CYLINDER

DATA_PATH = './test_data'
SWC_PATH = os.path.join(DATA_PATH, 'swc/')
fst_neuron = load_neuron(os.path.join(SWC_PATH, 'Neuron.swc'))


#def test_tree():
#    axes = []
#    for tree in fst_neuron.neurites:
#        fig, ax = view.tree(tree)
#        axes.append(ax)
#    nt.ok_(axes[0].get_data_ratio() > 1.00)
#    nt.ok_(axes[1].get_data_ratio() > 0.80)
#    nt.ok_(axes[2].get_data_ratio() > 1.00)
#    nt.ok_(axes[3].get_data_ratio() > 0.85)
#    tree0 = fst_neuron.neurites[0]
#    fig, ax = view.tree(tree0, treecolor='black', diameter=False, alpha=1., linewidth=1.2)
#    c = ax.collections[0]
#    nt.eq_(c.get_linewidth()[0], 1.2)
#    np.testing.assert_allclose(c.get_color(), np.array([[0., 0., 0., 1.]]))
#
#    nt.assert_raises(AssertionError, view.tree, tree0, plane='wrong')
#
#    common.plt.close('all')
#
#
#def test_neuron():
#    fig, ax = view.neuron(fst_neuron)
#    np.testing.assert_allclose(ax.get_xlim(), (-70.328535157399998, 94.7472627179))
#    np.testing.assert_allclose(ax.get_ylim(), (-87.600171997199993, 78.51626225230001))
#    nt.ok_(ax.get_title() == fst_neuron.name)
#
#    fig, ax = view.neuron(fst_neuron, xlim=(0, 100), ylim=(0, 100))
#    np.testing.assert_allclose(ax.get_xlim(), (0, 100))
#    np.testing.assert_allclose(ax.get_ylim(), (0, 100))
#    nt.ok_(ax.get_title() == fst_neuron.name)
#
#    nt.assert_raises(AssertionError, view.tree, fst_neuron, plane='wrong')
#
#    common.plt.close('all')
#
#
#def test_tree3d():
#    axes = []
#    for tree in fst_neuron.neurites:
#        fig, ax = view.tree3d(tree)
#        axes.append(ax)
#    nt.ok_(axes[0].get_data_ratio() > 1.00)
#    nt.ok_(axes[1].get_data_ratio() > 0.80)
#    nt.ok_(axes[2].get_data_ratio() > 1.00)
#    nt.ok_(axes[3].get_data_ratio() > 0.85)
#
#
#def test_neuron3d():
#    fig, ax = view.neuron3d(fst_neuron)
#    np.testing.assert_allclose(ax.get_xlim(), (-70.32853516, 94.74726272))
#    np.testing.assert_allclose(ax.get_ylim(), (-87.60017200, 78.51626225))
#    np.testing.assert_allclose(ax.get_zlim(), (-30.00000000, 84.20408797))
#    nt.ok_(ax.get_title() == fst_neuron.name)
#
#
#def test_neuron_no_neurites():
#    filename = os.path.join(SWC_PATH, 'point_soma.swc')
#    f, a = view.neuron(load_neuron(filename))
#
#
#def test_neuron3d_no_neurites():
#    filename = os.path.join(SWC_PATH, 'point_soma.swc')
#    f, a = view.neuron3d(load_neuron(filename))
#
#
#def test_dendrogram():
#    fig, ax = view.dendrogram(fst_neuron)
#    np.testing.assert_allclose(ax.get_xlim(), (-11.46075159339, 80.591751611909999))
#
#
#def test_one_point_branch():
#    test_section = Section(points=np.array([[1., 1., 1., 0.5, 2, 1, 0]]))
#    for diameter, linewidth in it.product((True, False),
#                                          (0.0, 1.2)):
#        view.tree(test_section, diameter=diameter, linewidth=linewidth)
#        view.tree3d(test_section, diameter=diameter, linewidth=linewidth)
#
#
#soma0 = fst_neuron.soma
#
##upright, varying radius
#soma_2pt_normal_pts = np.array([
#    [0.0,   0.0,  0.0, 1.0,  1, 1, -1],
#    [0.0,  10.0,  0.0, 10.0, 1, 2,  1],
#])
#soma_2pt_normal = make_soma(soma_2pt_normal_pts, soma_class=SOMA_CYLINDER)
#
##upright, uniform radius, multiple cylinders
#soma_3pt_normal_pts = np.array([
#    [0.0, -10.0,  0.0, 10.0, 1, 1, -1],
#    [0.0,   0.0,  0.0, 10.0, 1, 2,  1],
#    [0.0,   10.0, 0.0, 10.0, 1, 3,  2],
#])
#soma_3pt_normal = make_soma(soma_3pt_normal_pts, soma_class=SOMA_CYLINDER)
#
##increasing radius, multiple cylinders
#soma_4pt_normal_pts = np.array([
#    [0.0,   0.0,     0.0, 1.0, 1, 1, -1],
#    [0.0,   -10.0,   0.0, 2.0, 1, 2, 1],
#    [0.0,   -10.0, -10.0, 3.0, 1, 3, 2],
#    [-10.0, -10.0, -10.0, 4.0, 1, 4, 3],
#])
#soma_4pt_normal_cylinder = make_soma(soma_4pt_normal_pts, soma_class=SOMA_CYLINDER)
#soma_4pt_normal_contour = make_soma(soma_4pt_normal_pts, soma_class=SOMA_CONTOUR)
#
#
#def test_soma():
#    for s in (soma0,
#              soma_2pt_normal,
#              soma_3pt_normal,
#              soma_4pt_normal_cylinder,
#              soma_4pt_normal_contour):
#        fig, ax = view.soma(s)
#        common.plt.close(fig)
#
#        fig, ax = view.soma(s, outline=False)
#        common.plt.close(fig)
#
#    nt.assert_raises(AssertionError, view.tree, soma0, plane='wrong')
#
#
#def test_soma3d():
#    fig, ax = view.soma3d(soma0)
#    np.testing.assert_allclose(ax.get_xlim(), (-0.1,  0.2))
#    np.testing.assert_allclose(ax.get_ylim(), ( 0.0,  0.3))
#    np.testing.assert_allclose(ax.get_zlim(), (-0.1,  0.1))
