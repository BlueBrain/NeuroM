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

''' NeuroM, lightweight and fast

Examples:

    Load a neuron

    >>> from neurom import fst
    >>> nrn = fst.load_neuron('some/data/path/morph_file.swc')

    Obtain some morphometrics

    >>> apical_seg_lengths = fst.get('segment_lengths', \
                                     nrn, neurite_type=fst.NeuriteType.apical_dendrite)
    >>> axon_sec_lengths = fst.get('section_lengths', \
                                   nrn, neurite_type=fst.NeuriteType.axon)

    Load neurons from a directory. This loads all SWC or HDF5 files it finds\
    and returns a list of neurons

    >>> import numpy as np  # For mean value calculation
    >>> nrns = fst.load_neurons('some/data/directory')
    >>> for nrn in nrns:
    ...     print 'mean section length', np.mean(fst.get('section_lengths', nrn))


'''

import numpy as _np
from ._io import load_neuron, load_neurons, Neuron
from . import _mm
from ..utils import deprecated
from ..core.types import NeuriteType
from ..core.types import NEURITES as NEURITE_TYPES
from ..analysis.morphmath import segment_radius as seg_rad


load_population = deprecated('Use load_neurons instead.',
                             fun_name='load_population')(load_neurons)


NEURITEFEATURES = {
    'total_length': lambda nrn, **kwargs: _as_neurons(lambda n, **kw:
                                                      sum(_mm.section_lengths(n, **kw)),
                                                      nrn, **kwargs),
    'section_lengths': _mm.section_lengths,
    'section_path_distances': _mm.section_path_lengths,
    'number_of_sections': lambda nrn, **kwargs: _as_neurons(_mm.n_sections, nrn, **kwargs),
    'number_of_sections_per_neurite': _mm.n_sections_per_neurite,
    'number_of_neurites': lambda nrn, **kwargs: _as_neurons(_mm.n_neurites, nrn, **kwargs),
    'section_branch_orders': _mm.section_branch_orders,
    'section_radial_distances': _mm.section_radial_distances,
    'local_bifurcation_angles': _mm.local_bifurcation_angles,
    'remote_bifurcation_angles': _mm.remote_bifurcation_angles,
    'partition': _mm.bifurcation_partitions,
    'number_of_segments': lambda nrn, **kwargs: _as_neurons(_mm.n_segments, nrn, **kwargs),
    'trunk_origin_radii': _mm.trunk_origin_radii,
    'trunk_section_lengths': _mm.trunk_section_lengths,
    'segment_lengths': _mm.segment_lengths,
    'segment_radii': lambda nrn, **kwargs: [seg_rad(s) for s in _mm.iter_segments(nrn, **kwargs)],
    'segment_radial_distances': _mm.segment_radial_distances,
    'principal_direction_extents': _mm.principal_direction_extents
}

NEURONFEATURES = {
    'soma_radii': _mm.soma_radii,
    'soma_surface_areas': _mm.soma_surface_areas,
}


def _as_neurons(fun, nrns, **kwargs):
    '''Get features per neuron'''
    nrns = nrns.neurons if hasattr(nrns, 'neurons') else (nrns,)
    return [fun(n, **kwargs) for n in nrns]


def get(feature, *args, **kwargs):
    '''Neuron feature getter helper

    Returns features as a 1D numpy array.
    '''
    feature = (NEURITEFEATURES[feature] if feature in NEURITEFEATURES
               else NEURONFEATURES[feature])
    return _np.array(feature(*args, **kwargs))
