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
from neurom.features import neurite_features as _neuf
from neurom.features import neuron_features as _nrnf


NEURITEFEATURES = {
    'section_lengths': _neuf.section_lengths,
    'section_areas': _neuf.section_areas,
    'section_volumes': _neuf.section_volumes,
    'section_path_distances': _neuf.section_path_distances,
    'number_of_sections': _neuf.number_of_sections,
    'number_of_sections_per_neurite': _neuf.number_of_sections_per_neurite,
    'number_of_neurites': _neuf.neurite_number,
    'section_branch_orders': _neuf.section_branch_orders,
    'section_radial_distances': _neuf.section_radial_distances,
    'local_bifurcation_angles': _neuf.local_bifurcation_angles,
    'remote_bifurcation_angles': _neuf.remote_bifurcation_angles,
    'bifurcation_number': _neuf.bifurcation_number,
    'segment_lengths': _neuf.segment_lengths,
    'number_of_segments': _neuf.number_of_segments,
    'segment_taper_rates': _neuf.segment_taper_rates,
    'segment_radii': _neuf.segment_radii,
    'segment_x_coordinates': _neuf.segment_x_coordinates,
    'segment_y_coordinates': _neuf.segment_y_coordinates,
    'segment_z_coordinates': _neuf.segment_z_coordinates,
    'trunk_origin_radii': _neuf.trunk_origin_radii,
    'trunk_section_lengths': _neuf.trunk_section_lengths,
    'partition': _neuf.partition,
    'principal_direction_extents': _neuf.principal_directions_extents,
    'total_length_per_neurite': _neuf.total_length_per_neurite,
    'total_length': _neuf.total_length
}

NEURONFEATURES = {
    'soma_radii': _nrnf.soma_radii,
    'soma_surface_areas': _nrnf.soma_surface_areas,
    'trunk_origin_elevations': _nrnf.trunk_origin_elevations,
    'trunk_origin_azimuths': _nrnf.trunk_origin_azimuths
}


def get_feature(feature, *args, **kwargs):
    '''Neuron feature getter helper

    Returns features as a 1D numpy array.
    '''
    feature = (NEURITEFEATURES[feature] if feature in NEURITEFEATURES
               else NEURONFEATURES[feature])
    return _np.fromiter(feature(*args, **kwargs), _np.float)
