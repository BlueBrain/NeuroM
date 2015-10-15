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

'''Test neurom.stats

Since the stats module consists of simple wrappers to scipy.stats functions,
these tests are only sanity checks.
'''

from neurom import stats as st
from nose import tools as nt
import numpy as np
import random

np.random.seed(42)

NORMAL_MU = 10.
NORMAL_SIGMA = 1.0
NORMAL = np.random.normal(NORMAL_MU, NORMAL_SIGMA, 1000)

EXPON_LAMBDA = 10.
EXPON = np.random.exponential(EXPON_LAMBDA, 1000)

UNIFORM_MIN = -1.
UNIFORM_MAX = 1.
UNIFORM = np.random.uniform(UNIFORM_MIN, UNIFORM_MAX, 1000)

def test_fit_normal_params():
    params, errs = st.fit(NORMAL, 'norm')
    nt.assert_almost_equal(params[0], NORMAL_MU, 1)
    nt.assert_almost_equal(params[1], NORMAL_SIGMA, 1)


def test_fit_normal_regression():
    params, errs = st.fit(NORMAL, 'norm')
    nt.assert_almost_equal(params[0], 10.019332055822, 12)
    nt.assert_almost_equal(params[1], 0.978726207747, 12)
    nt.assert_almost_equal(errs[0], 0.021479979161, 12)
    nt.assert_almost_equal(errs[1], 0.745431659944, 12)


def test_fit_default_is_normal():
    p0, e0 = st.fit(NORMAL)
    p1, e1 = st.fit(NORMAL, 'norm')
    nt.assert_items_equal(p0, p1)
    nt.assert_items_equal(e0, e1)


def test_optimal_distribution_normal():
    optimal, params = st.optimal_distribution(NORMAL)
    nt.ok_(optimal=='norm')


def test_optimal_distribution_exponential():
    optimal, params = st.optimal_distribution(EXPON)
    nt.ok_(optimal=='expon')


def test_optimal_distribution_uniform():
    optimal, params = st.optimal_distribution(UNIFORM)
    nt.ok_(optimal=='uniform')
