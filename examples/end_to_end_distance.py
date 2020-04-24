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

"""Calculate and plot end-to-end distance of neurites."""

import neurom as nm
from neurom import morphmath
import numpy as np
import matplotlib.pyplot as plt


def path_end_to_end_distance(neurite):
    """Calculate and return end-to-end-distance of a given neurite."""
    trunk = neurite.root_node.points[0]
    return max(morphmath.point_dist(l.points[-1], trunk)
               for l in neurite.root_node.ileaf())


def mean_end_to_end_dist(neurites):
    """Calculate mean end to end distance for set of neurites."""
    return np.mean([path_end_to_end_distance(n) for n in neurites])


def make_end_to_end_distance_plot(nb_segments, end_to_end_distance, neurite_type):
    """Plot end-to-end distance vs number of segments."""
    plt.figure()
    plt.plot(nb_segments, end_to_end_distance)
    plt.title(neurite_type)
    plt.xlabel('Number of segments')
    plt.ylabel('End-to-end distance')
    plt.show()


def calculate_and_plot_end_to_end_distance(neurite):
    """Calculate and plot the end-to-end distance vs the number of segments for
    an increasingly larger part of a given neurite.

    Note that the plots are not very meaningful for bifurcating trees."""
    def _dist(seg):
        """Distance between segmenr end and trunk."""
        return morphmath.point_dist(seg[1], neurite.root_node.points[0])

    end_to_end_distance = [_dist(s) for s in nm.iter_segments(neurite)]
    make_end_to_end_distance_plot(np.arange(len(end_to_end_distance)) + 1,
                                  end_to_end_distance, neurite.type)


if __name__ == '__main__':
    #  load a neuron from an SWC file
    filename = 'test_data/swc/Neuron_3_random_walker_branches.swc'
    nrn = nm.load_neuron(filename)

    # print mean end-to-end distance per neurite type
    print('Mean end-to-end distance for axons: ',
          mean_end_to_end_dist(n for n in nrn.neurites if n.type == nm.AXON))
    print('Mean end-to-end distance for basal dendrites: ',
          mean_end_to_end_dist(n for n in nrn.neurites if n.type == nm.BASAL_DENDRITE))
    print('Mean end-to-end distance for apical dendrites: ',
          mean_end_to_end_dist(n for n in nrn.neurites
                               if n.type == nm.APICAL_DENDRITE))

    print('End-to-end distance per neurite (nb segments, end-to-end distance, neurite type):')
    for nrte in nrn.neurites:
        # plot end-to-end distance for increasingly larger parts of neurite
        calculate_and_plot_end_to_end_distance(nrte)
        # print (number of segments, end-to-end distance, neurite type)
        print(sum(len(s.points) - 1 for s in nrte.root_node.ipreorder()),
              path_end_to_end_distance(nrte), nrte.type)
