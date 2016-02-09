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
from functools import wraps
from neurom.features import neurite_features as _neuf
from neurom.features import neuron_features as _nrnf


def make_iterable(f, iterable_type=_np.ndarray):
    '''Packaging decorator. The decorator from the decorator module
    preserves the function signature with exact arguments upon wrapping
    '''
    @wraps(f)
    def wrapped(obj, **kwargs):
        ''' Feature function
        '''
        result = f(obj, **kwargs)

        if iterable_type is None:
            return result
        elif iterable_type is _np.ndarray:
            return _np.fromiter(result, _np.float)
        elif iterable_type is list or iterable_type is tuple:
            return iterable_type(result)
        else:
            raise TypeError('Unknown iterable type')
    return wrapped


NEURITEFEATURES = {'section_lengths': make_iterable(_neuf.section_lengths),
                   'section_number': make_iterable(_neuf.section_number),
                   'local_bifurcation_angles': make_iterable(_neuf.local_bifurcation_angles),
                   'remote_bifurcation_angles': make_iterable(_neuf.remote_bifurcation_angles),
                   'segment_lengths': make_iterable(_neuf.segment_lengths),
                   'trunk_origin_radii': make_iterable(_neuf.trunk_origin_radii),
                   'trunk_section_lengths': make_iterable(_neuf.trunk_section_lengths),
                   'principal_direction_extents': make_iterable(_neuf.principal_directions_extents)}


NEURONFEATURES = {'soma_radius': make_iterable(_nrnf.soma_radius),
                  'soma_surface_area': make_iterable(_nrnf.soma_surface_area)}