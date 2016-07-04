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
from neurom.io.utils import load_neuron
from neurom.fst import load_neuron as load_fst_neuron
from neurom import viewer
from neurom.analysis.morphtree import set_tree_type
import os
from matplotlib import pyplot as plt


class Dummy(object):
    pass


_PWD = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_PWD, '../../test_data/swc')
MORPH_FILENAME = os.path.join(DATA_PATH, 'Neuron.swc')

fst_nrn = load_fst_neuron(MORPH_FILENAME)
nrn = load_neuron(MORPH_FILENAME, set_tree_type)


def test_draw_neuron():
    viewer.draw(nrn)
    viewer.draw(fst_nrn)
    plt.close('all')


def test_draw_neuron3d():
    viewer.draw(nrn, mode='3d')
    viewer.draw(fst_nrn, mode='3d')
    plt.close('all')


def test_draw_tree():
    viewer.draw(nrn.neurites[0])
    viewer.draw(fst_nrn.neurites[0])
    plt.close('all')


def test_draw_tree3d():
    viewer.draw(nrn.neurites[0], mode='3d')
    viewer.draw(fst_nrn.neurites[0], mode='3d')
    plt.close('all')


def test_draw_soma():
    viewer.draw(nrn.soma)
    viewer.draw(fst_nrn.soma)
    plt.close('all')


def test_draw_soma3d():
    viewer.draw(nrn.soma, mode='3d')
    viewer.draw(fst_nrn.soma, mode='3d')


def test_draw_dendrogram():
    viewer.draw(nrn, mode='dendrogram')
    plt.close('all')


@nt.raises(viewer.InvalidDrawModeError)
def test_invalid_draw_mode_raises():
    viewer.draw(nrn, mode='4d')


@nt.raises(viewer.NotDrawableError)
def test_invalid_object_raises():
    viewer.draw(Dummy())


@nt.raises(viewer.NotDrawableError)
def test_invalid_combo_raises():
    viewer.draw(nrn.soma, mode='dendrogram')
