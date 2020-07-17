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

"""Test neurom.stats

Since the stats module consists of simple wrappers to scipy.stats functions,
these tests are only sanity checks.
"""

import numpy as np
from neurom import stats as st
from nose import tools as nt

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
    fit_ = st.fit(NORMAL, 'norm')
    nt.assert_almost_equal(fit_.params[0], NORMAL_MU, 1)
    nt.assert_almost_equal(fit_.params[1], NORMAL_SIGMA, 1)

def test_fit_normal_dict():
    fit_ = st.fit(NORMAL, 'norm')
    d = st.fit_results_to_dict(fit_, min_bound=-123, max_bound=123)
    nt.assert_almost_equal(d['mu'], NORMAL_MU, 1)
    nt.assert_almost_equal(d['sigma'], NORMAL_SIGMA, 1)
    nt.assert_almost_equal(d['min'], -123, 1)
    nt.assert_almost_equal(d['max'], 123, 1)

def test_fit_normal_regression():
    fit_ = st.fit(NORMAL, 'norm')
    nt.assert_almost_equal(fit_.params[0], 10.019332055822, 12)
    nt.assert_almost_equal(fit_.params[1], 0.978726207747, 12)
    nt.assert_almost_equal(fit_.errs[0], 0.021479979161, 12)
    nt.assert_almost_equal(fit_.errs[1], 0.7369569123250506, 12)

def test_fit_default_is_normal():
    fit0_ = st.fit(NORMAL)
    fit1_ = st.fit(NORMAL, 'norm')
    nt.eq_(fit0_.params, fit1_.params)
    nt.eq_(fit0_.errs, fit1_.errs)


def test_optimal_distribution_normal():
    optimal = st.optimal_distribution(NORMAL)
    nt.ok_(optimal.type == 'norm')


def test_optimal_distribution_exponential():
    optimal = st.optimal_distribution(EXPON)
    nt.ok_(optimal.type == 'expon')


def test_optimal_distribution_uniform():
    optimal = st.optimal_distribution(UNIFORM)
    nt.ok_(optimal.type == 'uniform')


def test_fit_results_dict_uniform():
    a = st.FitResults(params=[1, 2], errs=[3,4], type='uniform')
    d = st.fit_results_to_dict(a)
    nt.assert_equal(d['min'], 1)
    nt.assert_equal(d['max'], 3)
    nt.assert_equal(d['type'], 'uniform')

def test_fit_results_dict_uniform_min_max():
    a = st.FitResults(params=[1, 2], errs=[3,4], type='uniform')
    d = st.fit_results_to_dict(a, min_bound=-100, max_bound=100)
    nt.assert_equal(d['min'], 1)
    nt.assert_equal(d['max'], 3)
    nt.assert_equal(d['type'], 'uniform')


def test_fit_results_dict_normal():
    a = st.FitResults(params=[1, 2], errs=[3,4], type='norm')
    d = st.fit_results_to_dict(a)
    nt.assert_equal(d['mu'], 1)
    nt.assert_equal(d['sigma'], 2)
    nt.assert_equal(d['type'], 'normal')


def test_fit_results_dict_normal_min_max():
    a = st.FitResults(params=[1, 2], errs=[3,4], type='norm')
    d = st.fit_results_to_dict(a, min_bound=-100, max_bound=100)
    nt.assert_equal(d['mu'], 1)
    nt.assert_equal(d['sigma'], 2)
    nt.assert_equal(d['min'], -100)
    nt.assert_equal(d['max'], 100)
    nt.assert_equal(d['type'], 'normal')


def test_fit_results_dict_exponential():
    a = st.FitResults(params=[2, 2], errs=[3,4], type='expon')
    d = st.fit_results_to_dict(a)
    nt.assert_equal(d['lambda'], 1./2)
    nt.assert_equal(d['type'], 'exponential')


def test_fit_results_dict_exponential_min_max():
    a = st.FitResults(params=[2, 2], errs=[3,4], type='expon')
    d = st.fit_results_to_dict(a, min_bound=-100, max_bound=100)
    nt.assert_equal(d['lambda'], 1./2)
    nt.assert_equal(d['min'], -100)
    nt.assert_equal(d['max'], 100)
    nt.assert_equal(d['type'], 'exponential')

def test_scalar_stats():

    data = np.array([1.,2.,3.,4.,5.])

    result = st.scalar_stats(data)

    RESULT = {'mean': 3.,
              'max': 5.,
              'min': 1.,
              'std': 1.4142135623730951}

    nt.assert_true(RESULT == result)

def test_compare_two():
    data = np.array([1., 1., 2., 2.])
    data_same = np.array([1.0, 1.0, 2.0, 2.0])
    data_close = np.array([1.02, 1.01, 2.001, 2.0003])
    data_far = np.array([200., 100., 201])

    results1 = st.compare_two(data, data_same, test=st.StatTests.ks)
    nt.assert_almost_equal(results1.dist, 0.0)
    nt.assert_almost_equal(results1.pvalue, 1.0)

    results2 = st.compare_two(data, data_close, test=st.StatTests.ks)
    nt.assert_almost_equal(results2.dist, 0.5)

    results3 = st.compare_two(data, data_far, test=st.StatTests.ks)
    nt.assert_almost_equal(results3.dist, 1.0)

distr1 = np.ones(100)
distr2 = 2*np.ones(100)

def test_compare_two_ks():

    results1 = st.compare_two(distr1, distr1, test=st.StatTests.ks)
    nt.assert_almost_equal(results1.dist, 0.0, places=5)
    nt.assert_almost_equal(results1.pvalue, 1.0, places=5)

    results2 = st.compare_two(distr1, distr2, test=st.StatTests.ks)
    nt.assert_almost_equal(results2.dist, 1.0, places=5)
    nt.assert_almost_equal(results2.pvalue, 0.0, places=5)

def test_compare_two_wilcoxon():

    results2 = st.compare_two(distr1, distr2, test=st.StatTests.wilcoxon)
    nt.assert_almost_equal(results2.dist, 0.0, places=5)
    nt.assert_almost_equal(results2.pvalue, 0.0, places=5)

def test_compare_two_ttest():

    results1 = st.compare_two(distr1, distr1, test=st.StatTests.ttest)
    nt.ok_(np.isnan(results1.dist))
    nt.ok_(np.isnan(results1.pvalue))
    results2 = st.compare_two(distr1, distr2, test=st.StatTests.ttest)
    nt.ok_(np.isinf(results2.dist))
    nt.assert_almost_equal(results2.pvalue, 0.0, places=5)


@nt.raises(TypeError)
def test_compare_two_error():
    data = np.array([1., 1., 2., 2.])
    data_same = np.array([1.0, 1.0, 2.0, 2.0])
    results1 = st.compare_two(data, data_same, test='test')

def test_total_score():

    testList1 = (([1.,1., 1],[1.,1.,1.]),
                ([2.,3.,4.,5.],[2.,3.,4.,5.]))

    score = st.total_score(testList1)
    nt.assert_almost_equal(score, 0.)

    testList2 = (([1.,1., 1],[2.,2.,2.]),
                ([2.,3.,4.,5.],[2.,3.,4.,5.]))

    score = st.total_score(testList2, p=1)
    nt.assert_almost_equal(score, 1.)

    testList3 = (([1.,1., 1],[2.,2.,2.]),
                ([3.,3.,3.,3.],[4., 4., 4., 4.]))

    score = st.total_score(testList3, p=2)
    nt.assert_almost_equal(score, np.sqrt(2.))
