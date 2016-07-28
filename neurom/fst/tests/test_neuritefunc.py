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

'''Test neurom._neuritefunc functionality'''

from nose import tools as nt
import os
import numpy as np
from neurom import fst
from neurom.fst import _neuritefunc as _nf
from neurom.point_neurite.io import utils as io_utils
from neurom.core import tree as tr
from neurom.point_neurite import point_tree as ptr

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


def test_iter_segments():
    def seg_fun(seg):
        return seg[1][:4] - seg[0][:4]

    def seg_fun2(seg):
        return seg[1].value[:4] - seg[0].value[:4]

    a = np.array([seg_fun(s) for s in _nf.iter_segments(NRN)])
    b = np.array([seg_fun2(s) for s in tr.i_chain2(NRN_OLD.neurites, ptr.isegment)])

    _equal(a, b, debug=False)


def test_principal_direction_extents():
    # test with a realistic neuron
    nrn = fst.load_neuron(os.path.join(H5_PATH, 'bio_neuron-000.h5'))

    p_ref = [1672.9694359427331, 142.43704397865031, 226.45895382204986,
             415.50612748523838, 429.83008974193206, 165.95410536922873,
             346.83281498399697]

    p = _nf.principal_direction_extents(nrn)
    _close(np.array(p), np.array(p_ref))


s0 = fst.Section(42)
s1 = s0.add_child(fst.Section(42))
s2 = s0.add_child(fst.Section(42))
s3 = s0.add_child(fst.Section(42))
s4 = s1.add_child(fst.Section(42))
s5 = s1.add_child(fst.Section(42))
s6 = s4.add_child(fst.Section(42))
s7 = s4.add_child(fst.Section(42))


def test_n_bifurcation_points():
    nt.assert_equal(_nf.n_bifurcation_points(fst.Neurite(s0)), 2)
    nt.assert_equal(_nf.n_bifurcation_points(fst.Neurite(s1)), 2)
    nt.assert_equal(_nf.n_bifurcation_points(fst.Neurite(s2)), 0)
    nt.assert_equal(_nf.n_bifurcation_points(fst.Neurite(s3)), 0)
    nt.assert_equal(_nf.n_bifurcation_points(fst.Neurite(s4)), 1)
    nt.assert_equal(_nf.n_bifurcation_points(fst.Neurite(s5)), 0)
    nt.assert_equal(_nf.n_bifurcation_points(fst.Neurite(s6)), 0)
    nt.assert_equal(_nf.n_bifurcation_points(fst.Neurite(s7)), 0)


def test_n_forking_points():
    nt.assert_equal(_nf.n_forking_points(fst.Neurite(s0)), 3)
    nt.assert_equal(_nf.n_forking_points(fst.Neurite(s1)), 2)
    nt.assert_equal(_nf.n_forking_points(fst.Neurite(s2)), 0)
    nt.assert_equal(_nf.n_forking_points(fst.Neurite(s3)), 0)
    nt.assert_equal(_nf.n_forking_points(fst.Neurite(s4)), 1)
    nt.assert_equal(_nf.n_forking_points(fst.Neurite(s5)), 0)
    nt.assert_equal(_nf.n_forking_points(fst.Neurite(s6)), 0)
    nt.assert_equal(_nf.n_forking_points(fst.Neurite(s7)), 0)


def test_n_leaves():
    nt.assert_equal(_nf.n_leaves(fst.Neurite(s0)), 5)
    nt.assert_equal(_nf.n_leaves(fst.Neurite(s1)), 3)
    nt.assert_equal(_nf.n_leaves(fst.Neurite(s2)), 1)
    nt.assert_equal(_nf.n_leaves(fst.Neurite(s3)), 1)
    nt.assert_equal(_nf.n_leaves(fst.Neurite(s4)), 2)
    nt.assert_equal(_nf.n_leaves(fst.Neurite(s5)), 1)
    nt.assert_equal(_nf.n_leaves(fst.Neurite(s6)), 1)
    nt.assert_equal(_nf.n_leaves(fst.Neurite(s7)), 1)
