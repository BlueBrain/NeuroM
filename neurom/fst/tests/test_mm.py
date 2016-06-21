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

'''Test neurom._mm functionality'''

from nose import tools as nt
import os
import numpy as np
from neurom import fst
from neurom.fst import _mm
from neurom.analysis import morphmath as mmth
from neurom.io import utils as io_utils
from neurom.core import tree as tr
from neurom.core.neuron import make_soma
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


def test_bounding_box():

    ref_bboxes = [
        [[0.0, -52.79933167, 0.], [64.74726105, 0, 54.2040863]],
        [[-40.32853699, 0., 0.], [0.,45.98951721, 51.6414299 ]],
        [[0., 0., 0.], [64.24953461, 48.51626205, 50.51401901]],
        [[-33.25305939, -57.60017014, 0.], [0., 0., 49.70138168]]
    ]

    bboxes = [_mm.bounding_box(n) for n in NRN.neurites]
    nt.assert_true(np.allclose(bboxes, ref_bboxes))

def test_iter_segments():
    def seg_fun(seg):
        return seg[1][:4] - seg[0][:4]

    def seg_fun2(seg):
        return seg[1].value[:4] - seg[0].value[:4]

    a = np.array([seg_fun(s) for s in _mm.iter_segments(NRN)])
    b = np.array([seg_fun2(s) for s in tr.i_chain2(NRN_OLD.neurites, tr.isegment)])

    _equal(a, b, debug=False)


def test_section_tortuosity():

    sec_a = [
        (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)
    ]

    sec_b = [
        (0, 0, 0), (1, 0, 0), (1, 2, 0), (0, 2, 0)
    ]

    nt.eq_(_mm.section_tortuosity(sec_a), 1.0)
    nt.eq_(_mm.section_tortuosity(sec_b), 4.0 / 2.0)

    for s in _mm.iter_sections(NRN):
        nt.eq_(_mm.section_tortuosity(s),
               mmth.section_length(s) / mmth.point_dist(s[0], s[-1]))


def test_principal_direction_extents():
    # test with a realistic neuron
    nrn = fst.load_neuron(os.path.join(H5_PATH, 'bio_neuron-000.h5'))

    p_ref = [1672.9694359427331, 142.43704397865031, 226.45895382204986,
             415.50612748523838, 429.83008974193206, 165.95410536922873,
             346.83281498399697]

    p = _mm.principal_direction_extents(nrn)
    _close(np.array(p), np.array(p_ref))


def test_trunk_origin_elevations():
    class Mock(object):
        pass

    n0 = Mock()
    n1 = Mock()

    s = make_soma([[0, 0, 0, 4]])
    t0 = tr.Tree(((1, 0, 0, 2), (2, 1, 1, 2)))
    t0.type = fst.NeuriteType.basal_dendrite
    t1 = tr.Tree(((0, 1, 0, 2), (1, 2, 1, 2)))
    t1.type = fst.NeuriteType.basal_dendrite
    n0.neurites = [t0, t1]
    n0.soma = s

    t2 = tr.Tree(((0, -1, 0, 2), (-1, -2, -1, 2)))
    t2.type = fst.NeuriteType.basal_dendrite
    n1.neurites = [t2]
    n1.soma = s

    pop = Population([n0, n1])
    nt.assert_items_equal(_mm.trunk_origin_elevations(pop),
                          [0.0, np.pi/2., -np.pi/2.])

    nt.assert_items_equal(
        _mm.trunk_origin_elevations(pop, neurite_type=fst.NeuriteType.basal_dendrite),
        [0.0, np.pi/2., -np.pi/2.]
    )

    nt.eq_(
        len(_mm.trunk_origin_elevations(pop,
                                        neurite_type=fst.NeuriteType.axon)),
        0
    )

    nt.eq_(
        len(_mm.trunk_origin_elevations(pop,
                                        neurite_type=fst.NeuriteType.apical_dendrite)),
        0
    )

@nt.raises(Exception)
def test_trunk_elevation_zero_norm_vector_raises():
    _mm.trunk_origin_elevations(NRN)
