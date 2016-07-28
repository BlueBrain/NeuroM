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

'''Test neurom._neuronfunc functionality'''

from nose import tools as nt
import os
import numpy as np
from neurom import fst
from neurom.fst import _neuronfunc as _nf
from neurom.point_neurite.io import utils as io_utils
from neurom.core.soma import make_soma
from neurom.core.population import Population

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


def test_trunk_origin_elevations():
    class Mock(object):
        pass

    n0 = Mock()
    n1 = Mock()

    s = make_soma([[0, 0, 0, 4]])
    t0 = fst.Section(((1, 0, 0, 2), (2, 1, 1, 2)))
    t0.type = fst.NeuriteType.basal_dendrite
    t1 = fst.Section(((0, 1, 0, 2), (1, 2, 1, 2)))
    t1.type = fst.NeuriteType.basal_dendrite
    n0.neurites = [fst.Neurite(t0), fst.Neurite(t1)]
    n0.soma = s

    t2 = fst.Section(((0, -1, 0, 2), (-1, -2, -1, 2)))
    t2.type = fst.NeuriteType.basal_dendrite
    n1.neurites = [fst.Neurite(t2)]
    n1.soma = s

    pop = Population([n0, n1])
    nt.assert_items_equal(_nf.trunk_origin_elevations(pop),
                          [0.0, np.pi/2., -np.pi/2.])

    nt.assert_items_equal(
        _nf.trunk_origin_elevations(pop, neurite_type=fst.NeuriteType.basal_dendrite),
        [0.0, np.pi/2., -np.pi/2.]
    )

    nt.eq_(
        len(_nf.trunk_origin_elevations(pop,
                                        neurite_type=fst.NeuriteType.axon)),
        0
    )

    nt.eq_(
        len(_nf.trunk_origin_elevations(pop,
                                        neurite_type=fst.NeuriteType.apical_dendrite)),
        0
    )

@nt.raises(Exception)
def test_trunk_elevation_zero_norm_vector_raises():
    _nf.trunk_origin_elevations(NRN)
