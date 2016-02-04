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

from enum import Enum as _Enum
from decorator import decorator as _dec
import numpy as _np


class NeuriteFeatures(_Enum):
    '''Neurite Features
    '''
    segment_lengths = 1
    section_number = 2
    per_neurite_section_number = 3
    section_lengths = 4
    section_path_distances = 5
    section_radial_distances = 6
    local_bifurcation_angles = 7
    remote_bifurcation_angles = 8
    neurite_number = 9


class NeuronFeatures(_Enum):
    '''Neuron Features
    '''
    soma_radius = 10
    soma_surface_area = 11


@_dec
def _pkg(func, *args, **kwargs):
    '''Packaging decorator. The decorator from the decorator module
    preserves the function signature with exact arguments upon wrapping
    '''
    return _np.fromiter(func(*args, **kwargs), _np.float)


def _dispatch_feature(feature_enum):
    '''
    Returns the function that corresponds to the respective feature
    '''
    import neurom.features.neurite_features as _neuf
    import neurom.features.neuron_features as _nrnf
    if isinstance(feature_enum, NeuriteFeatures):
        return getattr(_neuf, feature_enum.name)
    elif isinstance(feature_enum, NeuronFeatures):
        return getattr(_nrnf, feature_enum.name)
    else:
        raise TypeError("Uknown Enum type")


def feature_factory(feature_enum):
    '''Feature Factory
    '''
    return _pkg(_dispatch_feature(feature_enum))
