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

''' Functionality for Feature Extraction'''


import numpy as _np
from . import _impl


NEURITEFEATURES = {
    'section_lengths': _impl.section_lengths,
    'section_areas': _impl.section_areas,
    'section_volumes': _impl.section_volumes,
    'section_path_distances': _impl.section_path_distances,
    'number_of_sections': _impl.number_of_sections,
    'number_of_sections_per_neurite': _impl.number_of_sections_per_neurite,
    'number_of_neurites': _impl.neurite_number,
    'section_branch_orders': _impl.section_branch_orders,
    'segment_radial_distances': _impl.segment_radial_distances,
    'section_radial_distances': _impl.section_radial_distances,
    'local_bifurcation_angles': _impl.local_bifurcation_angles,
    'remote_bifurcation_angles': _impl.remote_bifurcation_angles,
    'bifurcation_number': _impl.bifurcation_number,
    'number_of_bifurcations': _impl.bifurcation_number,
    'segment_lengths': _impl.segment_lengths,
    'number_of_segments': _impl.number_of_segments,
    'segment_taper_rates': _impl.segment_taper_rates,
    'segment_radii': _impl.segment_radii,
    'segment_x_coordinates': _impl.segment_x_coordinates,
    'segment_y_coordinates': _impl.segment_y_coordinates,
    'segment_z_coordinates': _impl.segment_z_coordinates,
    'trunk_origin_radii': _impl.trunk_origin_radii,
    'trunk_section_lengths': _impl.trunk_section_lengths,
    'partition': _impl.partition,
    'principal_direction_extents': _impl.principal_directions_extents,
    'total_length_per_neurite': _impl.total_length_per_neurite,
    'total_length': _impl.total_length
}

NEURONFEATURES = {
    'soma_radii': _impl.soma_radii,
    'soma_surface_areas': _impl.soma_surface_areas,
    'trunk_origin_elevations': _impl.trunk_origin_elevations,
    'trunk_origin_azimuths': _impl.trunk_origin_azimuths
}


def get(feature, *args, **kwargs):
    '''Neuron feature getter helper

    Returns features as a 1D numpy array.
    '''
    feature = (NEURITEFEATURES[feature] if feature in NEURITEFEATURES
               else NEURONFEATURES[feature])
    return _np.fromiter(feature(*args, **kwargs), _np.float)
