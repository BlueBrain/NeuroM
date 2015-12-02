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

from nose import tools as nt
from neurom.io.utils import make_neuron
from neurom.io.utils import load_neuron
from neurom import io
from neurom.view import view
from neurom.analysis.morphtree import find_tree_type
import os
import numpy as np
import pylab as plt
from neurom.core.tree import Tree


DATA_PATH = './test_data'
SWC_PATH = os.path.join(DATA_PATH, 'swc/')

data = io.load_data(SWC_PATH + 'Neuron.swc')
neuron0 = make_neuron(data, find_tree_type)
soma0 = neuron0.soma


def test_tree():
    axes = []
    for tree in neuron0.neurites:
        fig, ax = view.tree(tree)
        axes.append(ax)
    nt.ok_(axes[0].get_data_ratio() > 1.00 )
    nt.ok_(axes[1].get_data_ratio() > 0.80 )
    nt.ok_(axes[2].get_data_ratio() > 1.00 )
    nt.ok_(axes[3].get_data_ratio() > 0.85 )
    tree0 = neuron0.neurites[0]
    fig, ax = view.tree(tree0, treecolor='black', diameter=False, alpha=1., linewidth=1.2)
    c = ax.collections[0]
    nt.ok_(c.get_linewidth()[0] == 1.2 )
    nt.ok_(np.allclose(c.get_color(), np.array([[ 0.,  0.,  0.,  1.]])) )
    fig, ax = view.tree(tree0, plane='wrong')
    nt.ok_(ax == 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.')
    plt.close('all')


def test_soma():
    fig, ax = view.soma(soma0)
    nt.ok_(np.allclose(ax.get_xlim(), (0.0, 0.12)) )
    nt.ok_(np.allclose(ax.get_ylim(), (0.0, 0.20)) )
    fig, ax = view.soma(soma0, outline=False)
    nt.ok_(np.allclose(ax.get_xlim(), (0.0, 1.0)) )
    nt.ok_(np.allclose(ax.get_ylim(), (0.0, 1.0)) )
    fig, ax = view.soma(soma0, plane='wrong')
    nt.ok_(ax == 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.')
    plt.close('all')


def test_neuron():
    fig, ax = view.neuron(neuron0)
    nt.ok_(np.allclose(ax.get_xlim(), (-70.328535157399998, 94.7472627179)) )
    nt.ok_(np.allclose(ax.get_ylim(), (-87.600171997199993, 78.51626225230001)) )
    fig, ax = view.neuron(neuron0, plane='wrong')
    nt.ok_(ax == 'No such plane found! Please select one of: xy, xz, yx, yz, zx, zy.')
    plt.close('all')


def test_tree3d():
    axes = []
    for tree in neuron0.neurites:
        fig, ax = view.tree3d(tree)
        axes.append(ax)
    nt.ok_(axes[0].get_data_ratio() > 1.00 )
    nt.ok_(axes[1].get_data_ratio() > 0.80 )
    nt.ok_(axes[2].get_data_ratio() > 1.00 )
    nt.ok_(axes[3].get_data_ratio() > 0.85 )


def test_soma3d():
    fig, ax = view.soma3d(soma0)
    nt.ok_(np.allclose(ax.get_xlim(), (-0.2,  0.2)) )
    nt.ok_(np.allclose(ax.get_ylim(), (-0.2,  0.2)) )
    nt.ok_(np.allclose(ax.get_zlim(), (-0.2,  0.2)) )


def test_neuron3d():
    fig, ax = view.neuron3d(neuron0)
    nt.ok_(np.allclose(ax.get_xlim(), (-70.32853516, 94.74726272)) )
    nt.ok_(np.allclose(ax.get_ylim(), (-87.60017200, 78.51626225)) )
    nt.ok_(np.allclose(ax.get_zlim(), (-30.00000000, 84.20408797)) )


def test_neuron_no_neurites():
    filename = os.path.join(SWC_PATH, 'point_soma.swc')
    f, a = view.neuron(load_neuron(filename))


def test_neuron3d_no_neurites():
    filename = os.path.join(SWC_PATH, 'point_soma.swc')
    f, a = view.neuron3d(load_neuron(filename))


def test_dendrogram():
    fig, ax = view.dendrogram(neuron0)
    nt.ok_(np.allclose(ax.get_xlim(), (-11.46075159339, 80.591751611909999)))


def test_one_point_branch_with_diameter():
    test_tree = Tree(np.array([1., 1., 1., 0.5, 2, 1, 0]))
    try:
        view.tree(test_tree, diameter=True)
        nt.ok_(True)
    except:
        nt.ok_(False)
