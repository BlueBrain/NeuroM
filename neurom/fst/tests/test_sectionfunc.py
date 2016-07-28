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

'''Test neurom.sectionfunc functionality'''

from nose import tools as nt
import os
import math
import numpy as np
from neurom import fst
from neurom.fst import sectionfunc as _sf
from neurom.fst import _neuritefunc as _nf
from neurom.fst import Section
from neurom.analysis import morphmath as mmth
from neurom.point_neurite.io import utils as io_utils

_PWD = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(_PWD, '../../../test_data/h5/v1/')
DATA_PATH = os.path.join(H5_PATH, 'Neuron.h5')

NRN = fst.load_neuron(DATA_PATH)
NRN_OLD = io_utils.load_neuron(DATA_PATH)


def _equal(a, b, debug=False):
    if debug:
        print '\na.shape: %s\nb.shape: %s\n' % (a.shape, b.shape)
        print '\na: %s\nb:%s\n' % (a, b)
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.alltrue(a == b))


def _close(a, b, debug=False):
    if debug:
        print '\na.shape: %s\nb.shape: %s\n' % (a.shape, b.shape)
        print '\na: %s\nb:%s\n' % (a, b)
        print '\na - b:%s\n' % (a - b)
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.allclose(a, b))


def test_section_tortuosity():

    sec_a = Section([
        (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)
    ])

    sec_b = Section([
        (0, 0, 0), (1, 0, 0), (1, 2, 0), (0, 2, 0)
    ])

    nt.eq_(_sf.section_tortuosity(sec_a), 1.0)
    nt.eq_(_sf.section_tortuosity(sec_b), 4.0 / 2.0)

    for s in _nf.iter_sections(NRN):
        nt.eq_(_sf.section_tortuosity(s),
               mmth.section_length(s.points) / mmth.point_dist(s.points[0],
                                                               s.points[-1]))


def test_section_meander_angles():

    s0 = Section(np.array([[0, 0, 0],
                           [1, 0, 0],
                           [2, 0, 0],
                           [3, 0, 0],
                           [4, 0, 0]]))

    nt.assert_equal(_sf.section_meander_angles(s0),
                    [math.pi, math.pi, math.pi])

    s1 = Section(np.array([[0, 0, 0],
                           [1, 0, 0],
                           [1, 1, 0],
                           [2, 1, 0],
                           [2, 2, 0]]))

    nt.assert_equal(_sf.section_meander_angles(s1),
                    [math.pi / 2, math.pi / 2, math.pi / 2])

    s2 = Section(np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 0, 2],
                           [0, 0, 0]]))

    nt.assert_equal(_sf.section_meander_angles(s2),
                    [math.pi, 0.])


def test_section_meander_angles_single_segment():

    s = Section(np.array([[0, 0, 0], [1, 1, 1]]))

    nt.assert_equal(len(_sf.section_meander_angles(s)), 0)
