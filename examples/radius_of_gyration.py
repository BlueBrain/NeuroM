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

'''Calculate radius of gyration of neurites.'''

from neurom.analysis import morphmath
from neurom import ezy
from neurom.core import tree
from neurom.core.dataformat import COLS
import numpy as np


def segment_centre_of_mass(seg):
    '''Calculate and return centre of mass of a segment.

    Calculated as centre of mass of conical frustum'''
    h = morphmath.segment_length(seg)
    r0 = seg[0][COLS.R]
    r1 = seg[1][COLS.R]
    num = r0 * r0 + 2 * r0 * r1 + 3 * r1 * r1
    denom = 4 * (r0 * r0 + r0 * r1 + r1 * r1)
    centre_of_mass_z_loc = num / denom
    return seg[0][0:3] + (centre_of_mass_z_loc / h) * (seg[1][0:3] - seg[0][0:3])


def neurite_centre_of_mass(neurite):
    '''Calculate and return centre of mass of a neurite.'''
    centre_of_mass = np.zeros(3)
    total_volume = 0
    for segment in tree.val_iter(tree.isegment(neurite)):
        seg_volume = morphmath.segment_volume(segment)
        centre_of_mass = centre_of_mass + seg_volume * segment_centre_of_mass(segment)
        total_volume += seg_volume
    return centre_of_mass / total_volume


def distance_sqr(point, seg):
    '''Calculate and return square Euclidian distance from given point to
    centre of mass of given segment.'''
    centre_of_mass = segment_centre_of_mass(seg)
    return sum(pow(np.subtract(point, centre_of_mass), 2))


def radius_of_gyration(neurite):
    '''Calculate and return radius of gyration of a given neurite.'''
    centre_mass = neurite_centre_of_mass(neurite)
    sum_sqr_distance = 0
    N = 0
    for segment in tree.val_iter(tree.isegment(neurite)):
        sum_sqr_distance = sum_sqr_distance + distance_sqr(centre_mass, segment)
        N += 1
    return np.sqrt(sum_sqr_distance / N)


def mean_rad_of_gyration(neurites):
    '''Calculate mean radius of gyration for set of neurites.'''
    return np.mean([radius_of_gyration(n) for n in neurites])


if __name__ == '__main__':
    #  load a neuron from an SWC file
    filename = 'test_data/swc/Neuron.swc'
    nrn = ezy.Neuron(filename)

    # for every neurite, print (number of segments, radius of gyration, neurite type)
    print([(sum(1 for _ in tree.isegment(nrte)),
            radius_of_gyration(nrte), nrte.type) for nrte in nrn.neurites])

    # print mean radius of gyration per neurite type
    print('Mean radius of gyration for axons: ',
          mean_rad_of_gyration(n for n in nrn.neurites if n.type == ezy.TreeType.axon))
    print('Mean radius of gyration for basal dendrites: ',
          mean_rad_of_gyration(n for n in nrn.neurites if n.type == ezy.TreeType.basal_dendrite))
    print('Mean radius of gyration for apical dendrites: ',
          mean_rad_of_gyration(n for n in nrn.neurites if n.type == ezy.TreeType.apical_dendrite))
