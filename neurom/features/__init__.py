# Copyright (c) 2020, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
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

"""NeuroM, lightweight and fast.

Examples:
    Obtain some morphometrics

    >>> ap_seg_len = features.get('segment_lengths', nrn, neurite_type=neurom.APICAL_DENDRITE)
    >>> ax_sec_len = features.get('section_lengths', nrn, neurite_type=neurom.AXON)
"""

from functools import partial, update_wrapper
import numpy as _np
from enum import Enum, auto

class Shape(Enum):
    OnePerNeuron = auto()
    OnePerNeurite = auto()
    OnePerSection = auto()
    OnePerSegment = auto()
    Any = auto()

from neurom.features import neuritefunc as _nrt
from neurom.features import neuronfunc as _nrn
from neurom.core import NeuriteType as _ntype
from neurom.core import iter_neurites as _ineurites
from neurom.core.types import tree_type_checker as _is_type
from neurom.exceptions import NeuroMError
from neurom.features.register import FEATURES, get, register_feature, register_neurite_feature



NEURITEFEATURES = FEATURES['NEURITEFEATURES']
NEURONFEATURES = FEATURES['NEURONFEATURES']
