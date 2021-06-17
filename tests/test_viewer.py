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
import sys
import tempfile
from pathlib import Path

import matplotlib
import mock

if 'DISPLAY' not in os.environ:  # noqa
    matplotlib.use('Agg')  # noqa

from neurom import NeuriteType, load_neuron, viewer
from neurom.view import matplotlib_utils

import pytest
from numpy.testing import assert_allclose

DATA_PATH = Path(__file__).parent / 'data/swc'
MORPH_FILENAME = DATA_PATH / 'Neuron.swc'
nrn = load_neuron(MORPH_FILENAME)


def test_draw_neuron():
    viewer.draw(nrn)
    matplotlib_utils.plt.close('all')

    viewer.draw(nrn, realistic_diameters=True)
    matplotlib_utils.plt.close('all')


def test_draw_filter_neurite():
    for mode in ['2d', '3d']:
        viewer.draw(nrn, mode=mode, neurite_type=NeuriteType.basal_dendrite)
        assert_allclose(matplotlib_utils.plt.gca().get_ylim(),
                        [-30., 78], atol=5)

    matplotlib_utils.plt.close('all')


def test_draw_neuron3d():
    viewer.draw(nrn, mode='3d')
    matplotlib_utils.plt.close('all')

    with pytest.raises(NotImplementedError):
        viewer.draw(nrn, mode='3d', realistic_diameters=True)


def test_draw_tree():
    viewer.draw(nrn.neurites[0])
    matplotlib_utils.plt.close('all')


def test_draw_tree3d():
    viewer.draw(nrn.neurites[0], mode='3d')
    matplotlib_utils.plt.close('all')


def test_draw_soma():
    viewer.draw(nrn.soma)
    matplotlib_utils.plt.close('all')


def test_draw_soma3d():
    viewer.draw(nrn.soma, mode='3d')
    matplotlib_utils.plt.close('all')


def test_draw_dendrogram():
    viewer.draw(nrn, mode='dendrogram')
    matplotlib_utils.plt.close('all')

    viewer.draw(nrn.neurites[0], mode='dendrogram')
    matplotlib_utils.plt.close('all')

def test_draw_dendrogram_empty_segment():
    neuron = load_neuron(DATA_PATH / 'empty_segments.swc')
    viewer.draw(neuron, mode='dendrogram')
    matplotlib_utils.plt.close('all')



def test_invalid_draw_mode_raises():
    with pytest.raises(viewer.InvalidDrawModeError):
        viewer.draw(nrn, mode='4d')


def test_invalid_object_raises():
    with pytest.raises(viewer.NotDrawableError):
        class Dummy:
            pass
        viewer.draw(Dummy())


def test_invalid_combo_raises():
    with pytest.raises(viewer.NotDrawableError):
        viewer.draw(nrn.soma, mode='dendrogram')


def test_writing_output():
    with tempfile.TemporaryDirectory() as folder:
        output_dir = Path(folder, 'subdir')
        viewer.draw(nrn, mode='2d', output_path=output_dir)
        assert (output_dir / 'Figure.png').is_file()
        matplotlib_utils.plt.close('all')
