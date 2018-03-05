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

'''Test neurom.neuronfunc functionality'''
import tempfile
from nose import tools as nt
import os
import numpy as np
from neurom import load_neuron, NeuriteType
from neurom.features import neuronfunc as _nf
from neurom.core import Neurite, Section
from neurom.core import _soma
from neurom.core.dataformat import POINT_TYPE
from neurom.core.population import Population

from utils import _close, _equal

_PWD = os.path.dirname(os.path.abspath(__file__))

H5_PATH = os.path.join(_PWD, '../../../test_data/h5/v1/')
NRN = load_neuron(os.path.join(H5_PATH, 'Neuron.h5'))

SWC_PATH = os.path.join(_PWD, '../../../test_data/swc')
SIMPLE = load_neuron(os.path.join(SWC_PATH, 'simple.swc'))



def test_soma_surface_area():
    ret = _nf.soma_surface_area(SIMPLE)
    nt.eq_(ret, 12.566370614359172)

def test_soma_surface_areas():
    ret = _nf.soma_surface_areas(SIMPLE)
    nt.eq_(ret, [12.566370614359172, ])

def test_soma_radii():
    ret = _nf.soma_radii(SIMPLE)
    nt.eq_(ret, [1., ])

def test_trunk_section_lengths():
    ret = set(_nf.trunk_section_lengths(SIMPLE))
    nt.eq_(ret, set([5.0, 4.0]))

def test_trunk_origin_radii():
    ret = _nf.trunk_origin_radii(SIMPLE)
    nt.eq_(ret, [1.0, 1.0])

def test_trunk_origin_azimuths():
    ret = _nf.trunk_origin_azimuths(SIMPLE)
    nt.eq_(ret, [0.0, 0.0])

@nt.raises(Exception)
def test_trunk_elevation_zero_norm_vector_raises():
    _nf.trunk_origin_elevations(NRN)


def test_sholl_crossings_simple():
    center = SIMPLE.soma.center
    radii = []
    nt.eq_(list(_nf.sholl_crossings(SIMPLE, center, radii=radii)),
           [])

    radii = [1.0]
    nt.eq_([2],
           list(_nf.sholl_crossings(SIMPLE, center, radii=radii)))

    radii = [1.0, 5.1]
    nt.eq_([2, 4],
           list(_nf.sholl_crossings(SIMPLE, center, radii=radii)))

    radii = [1., 4., 5.]
    nt.eq_([2, 4, 5],
           list(_nf.sholl_crossings(SIMPLE, center, radii=radii)))


def load_swc(string):
    with tempfile.NamedTemporaryFile(prefix='test_neuron_func', mode='w', suffix='.swc') as fd:
        fd.write(string)
        fd.flush()
        return load_neuron(fd.name)


def test_sholl_analysis_custom():
    #recreate morphs from Fig 2 of
    #http://dx.doi.org/10.1016/j.jneumeth.2014.01.016
    radii = np.arange(10, 81, 10)
    center = 0, 0, 0
    morph_A = load_swc('''\
 1 1   0  0  0 1. -1
 2 3   0  0  0 1.  1
 3 3  80  0  0 1.  2
 4 4   0  0  0 1.  1
 5 4 -80  0  0 1.  4''')
    nt.eq_(list(_nf.sholl_crossings(morph_A, center, radii=radii)),
           [2, 2, 2, 2, 2, 2, 2, 2])

    morph_B = load_swc('''\
 1 1   0   0  0 1. -1
 2 3   0   0  0 1.  1
 3 3  35   0  0 1.  2
 4 3  51  10  0 1.  3
 5 3  51   5  0 1.  3
 6 3  51   0  0 1.  3
 7 3  51  -5  0 1.  3
 8 3  51 -10  0 1.  3
 9 4 -35   0  0 1.  2
10 4 -51  10  0 1.  9
11 4 -51   5  0 1.  9
12 4 -51   0  0 1.  9
13 4 -51  -5  0 1.  9
14 4 -51 -10  0 1.  9
                       ''')
    nt.eq_(list(_nf.sholl_crossings(morph_B, center, radii=radii)),
           [2, 2, 2, 10, 10, 0, 0, 0])

    morph_C = load_swc('''\
 1 1   0   0  0 1. -1
 2 3   0   0  0 1.  1
 3 3  65   0  0 1.  2
 4 3  85  10  0 1.  3
 5 3  85   5  0 1.  3
 6 3  85   0  0 1.  3
 7 3  85  -5  0 1.  3
 8 3  85 -10  0 1.  3
 9 4  65   0  0 1.  2
10 4  85  10  0 1.  9
11 4  85   5  0 1.  9
12 4  85   0  0 1.  9
13 4  85  -5  0 1.  9
14 4  85 -10  0 1.  9
                       ''')
    nt.eq_(list(_nf.sholl_crossings(morph_C, center, radii=radii)),
           [2, 2, 2, 2, 2, 2, 10, 10])
    #from neurom.view import view
    #view.neuron(morph_C)[0].savefig('foo.png')
