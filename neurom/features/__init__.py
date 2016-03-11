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
from decorator import decorator as _decorator
from neurom.features import neurite_features as _neuf
from neurom.features import neuron_features as _nrnf


def make_iterable(iterable_type=_np.ndarray):
    '''Decorator factory. Dispatches the decorator that
    corresponds to the type of iterable which is given as
    an argument.
    '''
    @_decorator
    def wrapped(f, *args, **kwargs):
        ''' Feature function
        '''
        result = f(*args, **kwargs)
        if iterable_type is _np.ndarray:
            return _np.fromiter(result, _np.float)
        elif iterable_type is list or iterable_type is tuple:
            return iterable_type(result)
        else:
            raise TypeError('Unknown iterable type')
    return wrapped


NEURITEFEATURES = {
    'section_lengths': make_iterable()(_neuf.section_lengths),
    'section_areas': make_iterable()(_neuf.section_areas),
    'section_volumes': make_iterable()(_neuf.section_volumes),
    'section_path_distances': make_iterable()(_neuf.section_path_distances),
    'number_of_sections': make_iterable()(_neuf.number_of_sections),
    'number_of_sections_per_neurite': make_iterable()(_neuf.number_of_sections_per_neurite),
    'section_branch_orders': make_iterable()(_neuf.section_branch_orders),
    'section_radial_distances': make_iterable()(_neuf.section_radial_distances),
    'local_bifurcation_angles': make_iterable()(_neuf.local_bifurcation_angles),
    'remote_bifurcation_angles': make_iterable()(_neuf.remote_bifurcation_angles),
    'bifurcation_number': make_iterable()(_neuf.bifurcation_number),
    'segment_lengths': make_iterable()(_neuf.segment_lengths),
    'number_of_segments': make_iterable()(_neuf.number_of_segments),
    'segment_taper_rates': make_iterable()(_neuf.segment_taper_rates),
    'trunk_origin_radii': make_iterable()(_neuf.trunk_origin_radii),
    'trunk_section_lengths': make_iterable()(_neuf.trunk_section_lengths),
    'partition': make_iterable()(_neuf.partition),
    'principal_direction_extents': make_iterable()(_neuf.principal_directions_extents)
}


NEURONFEATURES = {'soma_radii': make_iterable()(_nrnf.soma_radii),
                  'soma_surface_areas': make_iterable()(_nrnf.soma_surface_areas),
                  'trunk_origin_elevations': make_iterable()(_nrnf.trunk_origin_elevations),
                  'trunk_origin_azimuths': make_iterable()(_nrnf.trunk_origin_azimuths)}
