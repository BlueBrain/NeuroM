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

'''Load and view multiple somas'''

from neurom import ezy
from neurom.view.common import plot_sphere
import matplotlib.pyplot as plt
import numpy as np


def random_color():
    '''Random color generation'''
    return np.random.rand(3, 1)


def load_neurons(file_names):
    '''Load neurons from list of file names'''
    neurons = []
    for file_name in file_names:
        neurons.append(ezy.Neuron(file_name))
    return neurons


def extract_somas(neurons):
    '''Extract and return somas from set of neurons'''
    somas = []
    for neuron in neurons:
        somas.append(neuron.soma)
    return somas


def plot_somas(neurons):
    '''Plot set of somas on same figure as spheres, each with different color'''
    somas = extract_somas(neurons)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    for s in somas:
        center = s.center
        radius = s.radius
        plot_sphere(fig, ax, center, radius, color=random_color(), alpha=1)
    plt.show()


if __name__ == '__main__':
    #  define set of files containing relevant neurons
    file_nms = []
    file_nms.append('test_data/swc/Soma_origin.swc')
    file_nms.append('test_data/swc/Soma_translated_1.swc')
    file_nms.append('test_data/swc/Soma_translated_2.swc')

    # load from file and plot
    plot_somas(load_neurons(file_nms))
