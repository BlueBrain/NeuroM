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

'''Easy analysis examples

These examples highlight most of the pre-packaged neurom.ezy.Neuron
morphometrics functionality.

'''

from __future__ import print_function
from pprint import pprint
from neurom import ezy
import numpy as np


def stats(data):
    '''Dictionary with summary stats for data

    Returns:
        dicitonary with length, mean, sum, standard deviation,\
            min and max of data
    '''
    return {'len': len(data),
            'mean': np.mean(data),
            'sum': np.sum(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)}


def pprint_stats(data):
    '''Pretty print summary stats for data'''
    pprint(stats(data))


if __name__ == '__main__':

    filename = 'test_data/swc/Neuron.swc'

    #  load a neuron from an SWC file
    nrn = ezy.load_neuron(filename)

    # Get some soma information
    # Soma radius and surface area
    print("Soma radius", nrn.get_soma_radius())
    print("Soma surface area", nrn.get_soma_surface_area())

    # Get information about neurites
    # Most neurite data can be queried for a particular type of neurite.
    # The allowed types are members of the TreeType enumeration.
    # NEURITE_TYPES is a list of valid neurite types.

    # We start by calling methods for different neurite types separately
    # to warm up...

    # number of neurites
    print('Number of neurites (all):', nrn.get_n_neurites())
    print('Number of neurites (axons):', nrn.get_n_neurites(ezy.TreeType.axon))
    print('Number of neurites (apical dendrites):',
          nrn.get_n_neurites(ezy.TreeType.apical_dendrite))
    print('Number of neurites (basal dendrites):',
          nrn.get_n_neurites(ezy.TreeType.basal_dendrite))

    # number of sections
    print('Number of sections:', nrn.get_n_sections())
    print('Number of sections (axons):', nrn.get_n_sections(ezy.TreeType.axon))
    print('Number of sections (apical dendrites):',
          nrn.get_n_sections(ezy.TreeType.apical_dendrite))
    print('Number of sections (basal dendrites):',
          nrn.get_n_sections(ezy.TreeType.basal_dendrite))

    # number of sections per neurite
    print('Number of sections per neurite:',
          nrn.get_n_sections_per_neurite())
    print('Number of sections per neurite (axons):',
          nrn.get_n_sections_per_neurite(ezy.TreeType.axon))
    print('Number of sections per neurite (apical dendrites):',
          nrn.get_n_sections_per_neurite(ezy.TreeType.apical_dendrite))
    print('Number of sections per neurite (basal dendrites):',
          nrn.get_n_sections_per_neurite(ezy.TreeType.basal_dendrite))

    # OK, this is getting repetitive, so lets loop over valid neurite types.
    # The following methods return arrays of measurements. We will gather some
    # summary statistics for each and print them.

    # Section lengths for all and different types of neurite
    for ttype in ezy.NEURITE_TYPES:
        sec_len = nrn.get_section_lengths(ttype)
        print('Section lengths (', ttype, '):', sep='')
        pprint_stats(sec_len)

    # Segment lengths for all and different types of neurite
    for ttype in ezy.NEURITE_TYPES:
        seg_len = nrn.get_segment_lengths(ttype)
        print('Segment lengths (', ttype, '):', sep='')
        pprint_stats(seg_len)

    # Section radial distances for all and different types of neurite
    # Careful! Here we need to pass tree type as a named argument
    for ttype in ezy.NEURITE_TYPES:
        sec_rad_dist = nrn.get_section_radial_distances(neurite_type=ttype)
        print('Section radial distance (', ttype, '):', sep='')
        pprint_stats(sec_rad_dist)

    # Section path distances for all and different types of neurite
    # Careful! Here we need to pass tree type as a named argument
    for ttype in ezy.NEURITE_TYPES:
        sec_path_dist = nrn.get_section_path_distances(neurite_type=ttype)
        print('Section path distance (', ttype, '):', sep='')
        pprint_stats(sec_path_dist)

    # Local bifurcation angles for all and different types of neurite
    for ttype in ezy.NEURITE_TYPES:
        local_bifangles = nrn.get_local_bifurcation_angles(ttype)
        print('Local bifurcation angles (', ttype, '):', sep='')
        pprint_stats(local_bifangles)

    # Remote bifurcation angles for all and different types of neurite
    for ttype in ezy.NEURITE_TYPES:
        rem_bifangles = nrn.get_remote_bifurcation_angles(ttype)
        print('Local bifurcation angles (', ttype, '):', sep='')
        pprint_stats(rem_bifangles)

    # Neuron's bounding box
    print('Bounding box ((min x, y, z), (max x, y, z))',
          nrn.bounding_box())
