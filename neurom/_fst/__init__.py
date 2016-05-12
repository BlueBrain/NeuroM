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
from .io import load_neuron
from . import mm
from ..core.types import NeuriteType


NEURITEFEATURES = {
    'section_lengths': mm.section_lengths,
    'section_path_distances': mm.path_lengths,
    'number_of_sections': lambda *args, **kwargs: [mm.n_sections(*args, **kwargs)],
    'number_of_sections_per_neurite': mm.n_sections_per_neurite,
    'number_of_neurites': lambda *args, **kwargs: [mm.n_neurites(*args, **kwargs)],
    'section_branch_orders': mm.section_branch_orders,
    'section_radial_distances': mm.section_radial_distances,
    'local_bifurcation_angles': mm.local_bifurcation_angles,
    'remote_bifurcation_angles': mm.remote_bifurcation_angles,
    'partition': mm.bifurcation_partitions,
    'number_of_segments': lambda *args, **kwargs: [mm.n_segments(*args, **kwargs)],
    'trunk_origin_radii': mm.trunk_origin_radii,
    'trunk_section_lengths': mm.trunk_section_lengths,
}

NEURONFEATURES = {
    'soma_radii': mm.soma_radii,
    'soma_surface_areas': mm.soma_surface_areas,
}


def get(feature, *args, **kwargs):
    '''Neuron feature getter helper

    Returns features as a 1D numpy array.
    '''
    feature = (NEURITEFEATURES[feature] if feature in NEURITEFEATURES
               else NEURONFEATURES[feature])
    return _np.array(feature(*args, **kwargs))
