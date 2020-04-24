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

"""Easy analysis examples

These examples highlight most of the pre-packaged neurom.nm.get
morphometrics functionality.

"""

from __future__ import print_function
from pprint import pprint
import numpy as np
import neurom as nm


def stats(data):
    """Dictionary with summary stats for data

    Returns:
        dicitonary with length, mean, sum, standard deviation,\
            min and max of data
    """
    return {'len': len(data),
            'mean': np.mean(data),
            'sum': np.sum(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)}


def pprint_stats(data):
    """Pretty print summary stats for data."""
    pprint(stats(data))


if __name__ == '__main__':

    filename = 'test_data/swc/Neuron.swc'

    #  load a neuron from an SWC file
    nrn = nm.load_neuron(filename)

    # Get some soma information
    # Soma radius and surface area
    print("Soma radius", nm.get('soma_radii', nrn)[0])
    print("Soma surface area", nm.get('soma_surface_areas', nrn)[0])

    # Get information about neurites
    # Most neurite data can be queried for a particular type of neurite.
    # The allowed types are members of the NeuriteType enumeration.
    # NEURITE_TYPES is a list of valid neurite types.

    # We start by calling methods for different neurite types separately
    # to warm up...

    # number of neurites
    print('Number of neurites (all):', nm.get('number_of_neurites', nrn)[0])
    print('Number of neurites (axons):',
          nm.get('number_of_neurites', nrn, neurite_type=nm.NeuriteType.axon)[0])
    print('Number of neurites (apical dendrites):',
          nm.get('number_of_neurites', nrn, neurite_type=nm.NeuriteType.apical_dendrite)[0])
    print('Number of neurites (basal dendrites):',
          nm.get('number_of_neurites', nrn, neurite_type=nm.NeuriteType.basal_dendrite)[0])

    # number of sections
    print('Number of sections:',
          nm.get('number_of_sections', nrn)[0])
    print('Number of sections (axons):',
          nm.get('number_of_sections', nrn, neurite_type=nm.NeuriteType.axon)[0])
    print('Number of sections (apical dendrites):',
          nm.get('number_of_sections', nrn, neurite_type=nm.NeuriteType.apical_dendrite)[0])
    print('Number of sections (basal dendrites):',
          nm.get('number_of_sections', nrn, neurite_type=nm.NeuriteType.basal_dendrite)[0])

    # number of sections per neurite
    print('Number of sections per neurite:',
          nm.get('number_of_sections_per_neurite', nrn))
    print('Number of sections per neurite (axons):',
          nm.get('number_of_sections_per_neurite', nrn, neurite_type=nm.NeuriteType.axon))
    print('Number of sections per neurite (apical dendrites):',
          nm.get('number_of_sections_per_neurite',
                 nrn, neurite_type=nm.NeuriteType.apical_dendrite))
    print('Number of sections per neurite (basal dendrites):',
          nm.get('number_of_sections_per_neurite',
                 nrn, neurite_type=nm.NeuriteType.apical_dendrite))

    # OK, this is getting repetitive, so lets loop over valid neurite types.
    # The following methods return arrays of measurements. We will gather some
    # summary statistics for each and print them.

    # Section lengths for all and different types of neurite
    for ttype in nm.NEURITE_TYPES:
        sec_len = nm.get('section_lengths', nrn, neurite_type=ttype)
        print('Section lengths (', ttype, '):', sep='')
        pprint_stats(sec_len)

    # Segment lengths for all and different types of neurite
    for ttype in nm.NEURITE_TYPES:
        seg_len = nm.get('segment_lengths', nrn, neurite_type=ttype)
        print('Segment lengths (', ttype, '):', sep='')
        pprint_stats(seg_len)

    # Section radial distances for all and different types of neurite
    # Careful! Here we need to pass tree type as a named argument
    for ttype in nm.NEURITE_TYPES:
        sec_rad_dist = nm.get('section_radial_distances', nrn, neurite_type=ttype)
        print('Section radial distance (', ttype, '):', sep='')
        pprint_stats(sec_rad_dist)

    # Section path distances for all and different types of neurite
    # Careful! Here we need to pass tree type as a named argument
    for ttype in nm.NEURITE_TYPES:
        sec_path_dist = nm.get('section_path_distances', nrn, neurite_type=ttype)
        print('Section path distance (', ttype, '):', sep='')
        pprint_stats(sec_path_dist)

    # Local bifurcation angles for all and different types of neurite
    for ttype in nm.NEURITE_TYPES:
        local_bifangles = nm.get('local_bifurcation_angles', nrn, neurite_type=ttype)
        print('Local bifurcation angles (', ttype, '):', sep='')
        pprint_stats(local_bifangles)

    # Remote bifurcation angles for all and different types of neurite
    for ttype in nm.NEURITE_TYPES:
        rem_bifangles = nm.get('remote_bifurcation_angles', nrn, neurite_type=ttype)
        print('Local bifurcation angles (', ttype, '):', sep='')
        pprint_stats(rem_bifangles)
