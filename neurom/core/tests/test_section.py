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

import math
from nose import tools as nt
import neurom as nm
from neurom.core import Section
import numpy as np


RADIUS = 4.
POINTS = np.array([[0., 0., 0., RADIUS],
                   [0., 0., 1., RADIUS],
                   [0., 0., 2., RADIUS],
                   [0., 0., 3., RADIUS],
                   [1., 0., 3., RADIUS],
                   [2., 0., 3., RADIUS],
                   [3., 0., 3., RADIUS]])

REF_LEN = 6.0


def test_init_empty():

    sec = Section([])
    nt.ok_(sec.id is None)
    nt.eq_(sec.type, nm.NeuriteType.undefined)
    nt.eq_(len(sec.points), 0)


def test_section_id():

    sec = Section([], section_id=42)
    nt.eq_(sec.id, 42)


def test_section_type():

    sec = Section([], section_type=nm.AXON)
    nt.eq_(sec.type, nm.AXON)

    sec = Section([], section_type=nm.BASAL_DENDRITE)
    nt.eq_(sec.type, nm.BASAL_DENDRITE)


def test_section_length():

    sec = Section(POINTS)
    nt.assert_almost_equal(sec.length, REF_LEN)


def test_section_area():

    sec = Section(POINTS)
    area = 2 * math.pi * RADIUS * REF_LEN
    nt.assert_almost_equal(sec.area, area)


def test_section_volume():

    sec = Section(POINTS)
    volume = math.pi * RADIUS * RADIUS * REF_LEN
    nt.assert_almost_equal(sec.volume, volume)



