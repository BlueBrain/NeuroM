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

"""Calculate radius of gyration of neurites."""

import neurom as nm
from neurom import morphmath as mm
from neurom.core.dataformat import COLS
import numpy as np


def segment_centre_of_mass(seg):
    """Calculate and return centre of mass of a segment.

    C, seg_volalculated as centre of mass of conical frustum"""
    h = mm.segment_length(seg)
    r0 = seg[0][COLS.R]
    r1 = seg[1][COLS.R]
    num = r0 * r0 + 2 * r0 * r1 + 3 * r1 * r1
    denom = 4 * (r0 * r0 + r0 * r1 + r1 * r1)
    centre_of_mass_z_loc = num / denom
    return seg[0][COLS.XYZ] + (centre_of_mass_z_loc / h) * (seg[1][COLS.XYZ] - seg[0][COLS.XYZ])


def neurite_centre_of_mass(neurite):
    """Calculate and return centre of mass of a neurite."""
    centre_of_mass = np.zeros(3)
    total_volume = 0

    seg_vol = np.array(map(mm.segment_volume, nm.iter_segments(neurite)))
    seg_centre_of_mass = np.array(map(segment_centre_of_mass, nm.iter_segments(neurite)))

    # multiply array of scalars with array of arrays
    # http://stackoverflow.com/questions/5795700/multiply-numpy-array-of-scalars-by-array-of-vectors
    seg_centre_of_mass = seg_centre_of_mass * seg_vol[:, np.newaxis]
    centre_of_mass = np.sum(seg_centre_of_mass, axis=0)
    total_volume = np.sum(seg_vol)
    return centre_of_mass / total_volume


def distance_sqr(point, seg):
    """Calculate and return square Euclidian distance from given point to
    centre of mass of given segment."""
    centre_of_mass = segment_centre_of_mass(seg)
    return sum(pow(np.subtract(point, centre_of_mass), 2))


def radius_of_gyration(neurite):
    """Calculate and return radius of gyration of a given neurite."""
    centre_mass = neurite_centre_of_mass(neurite)
    sum_sqr_distance = 0
    N = 0
    dist_sqr = [distance_sqr(centre_mass, s) for s in nm.iter_segments(neurite)]
    sum_sqr_distance = np.sum(dist_sqr)
    N = len(dist_sqr)
    return np.sqrt(sum_sqr_distance / N)


def mean_rad_of_gyration(neurites):
    """Calculate mean radius of gyration for set of neurites."""
    return np.mean([radius_of_gyration(n) for n in neurites])


if __name__ == '__main__':
    #  load a neuron from an SWC file
    filename = 'test_data/swc/Neuron.swc'
    nrn = nm.load_neuron(filename)

    # for every neurite, print (number of segments, radius of gyration, neurite type)
    print([(sum(len(s.points) - 1 for s in nrte.iter_sections()),
            radius_of_gyration(nrte), nrte.type) for nrte in nrn.neurites])

    # print mean radius of gyration per neurite type
    print('Mean radius of gyration for axons: ',
          mean_rad_of_gyration(n for n in nrn.neurites if n.type == nm.AXON))
    print('Mean radius of gyration for basal dendrites: ',
          mean_rad_of_gyration(n for n in nrn.neurites if n.type == nm.BASAL_DENDRITE))
    print('Mean radius of gyration for apical dendrites: ',
          mean_rad_of_gyration(n for n in nrn.neurites
                               if n.type == nm.APICAL_DENDRITE))
