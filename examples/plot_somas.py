#!/usr/bin/env python
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

"""Load and view multiple somas."""

from pathlib import Path

from neurom import load_neuron
import neurom.view.common as common
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path(__file__).parent.parent / 'test_data'
SWC_PATH = Path(DATA_PATH, 'swc')


def random_color():
    """Random color generation."""
    return np.random.rand(3, 1)


def plot_somas(somas):
    """Plot set of somas on same figure as spheres, each with different color."""
    _, ax = common.get_figure(new_fig=True, subplot=111,
                              params={'projection': '3d', 'aspect': 'equal'})
    for s in somas:
        common.plot_sphere(ax, s.center, s.radius, color=random_color(), alpha=1)
    plt.show()


if __name__ == '__main__':
    #  define set of files containing relevant neurons
    file_nms = [Path(SWC_PATH, file_nm) for file_nm in ['Soma_origin.swc',
                                                                'Soma_translated_1.swc',
                                                                'Soma_translated_2.swc']]

    # load from file and plot
    sms = [load_neuron(file_nm).soma for file_nm in file_nms]
    plot_somas(sms)
