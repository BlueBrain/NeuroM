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

''' NeuroM, lightweight and fast '''

import numpy as _np
from ._io import load_neuron, load_neurons, load_population, Neuron
from . import _mm
from ..core.types import NeuriteType


NEURITEFEATURES = {
    'total_length': lambda *args, **kwargs: [sum(_mm.section_lengths(*args, **kwargs))],
    'section_lengths': _mm.section_lengths,
    'section_path_distances': _mm.section_path_lengths,
    'number_of_sections': lambda *args, **kwargs: [_mm.n_sections(*args, **kwargs)],
    'number_of_sections_per_neurite': _mm.n_sections_per_neurite,
    'number_of_neurites': lambda *args, **kwargs: [_mm.n_neurites(*args, **kwargs)],
    'section_branch_orders': _mm.section_branch_orders,
    'section_radial_distances': _mm.section_radial_distances,
    'local_bifurcation_angles': _mm.local_bifurcation_angles,
    'remote_bifurcation_angles': _mm.remote_bifurcation_angles,
    'partition': _mm.bifurcation_partitions,
    'number_of_segments': lambda *args, **kwargs: [_mm.n_segments(*args, **kwargs)],
    'trunk_origin_radii': _mm.trunk_origin_radii,
    'trunk_section_lengths': _mm.trunk_section_lengths,
    'segment_lengths': _mm.segment_lengths,
    'segment_radial_distances': _mm.segment_radial_distances,
    'principal_direction_extents': _mm.principal_direction_extents
}

NEURONFEATURES = {
    'soma_radii': _mm.soma_radii,
    'soma_surface_areas': _mm.soma_surface_areas,
}


def get(feature, *args, **kwargs):
    '''Neuron feature getter helper

    Returns features as a 1D numpy array.
    '''
    feature = (NEURITEFEATURES[feature] if feature in NEURITEFEATURES
               else NEURONFEATURES[feature])
    return _np.array(feature(*args, **kwargs))
