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
"""Compatibility between NL and H5 files."""
# pylint: disable=protected-access

import numpy as np

import neurom as nm
from neurom.features import neurite as _nf

m_h5 = nm.load_morphology('tests/data/h5/v1/bio_neuron-001.h5')
m_asc = nm.load_morphology('tests/data/neurolucida/bio_neuron-001.asc')

print('h5 number of sections: %s' % nm.get('number_of_sections', m_h5)[0])
print('nl number of sections: %s\n' % nm.get('number_of_sections', m_asc)[0])
print('h5 number of segments: %s' % nm.get('number_of_segments', m_h5)[0])
print('nl number of segments: %s\n' % nm.get('number_of_segments', m_asc)[0])
print('h5 total neurite length: %s' %
      np.sum(nm.get('section_lengths', m_h5)))
print('nl total neurite length: %s\n' %
      np.sum(nm.get('section_lengths', m_asc)))
print('h5 principal direction extents: %s' %
      nm.get('principal_direction_extents', m_h5))
print('nl principal direction extents: %s' %
      nm.get('principal_direction_extents', m_asc))

print('\nNumber of neurites:')
for nt in iter(nm.NeuriteType):
    print(nt, _nf.n_neurites(m_h5, neurite_type=nt), _nf.n_neurites(m_asc, neurite_type=nt))

print('\nNumber of segments:')
for nt in iter(nm.NeuriteType):
    print(nt, _nf.n_segments(m_h5, neurite_type=nt), _nf.n_segments(m_asc, neurite_type=nt))
