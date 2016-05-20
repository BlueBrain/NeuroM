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

'''Statistical analysis helper functions

Nothing fancy. Just commonly used functions using scipy functionality.'''

from collections import namedtuple
from collections import OrderedDict
from scipy import stats as _st
import numpy as np
from enum import Enum, unique

FitResults = namedtuple('FitResults', ['params', 'errs', 'type'])


@unique
class StatTests(Enum):
    '''Enum representing valid statistical tests of scipy'''
    ks = 1
    wilcoxon = 2
    ttest = 3


def get_test(stest):
    '''Returns the correct stat test'''
    sts = {StatTests.ks: 'ks_2samp', StatTests.wilcoxon: 'wilcoxon', StatTests.ttest: 'ttest_ind'}

    if stest in StatTests:
        return sts[stest]
    else:
        raise TypeError('Statistical test not recognized. Choose from ks, wilcoxon, ttest.')


def fit_results_to_dict(fit_results, min_bound=None, max_bound=None):
    '''Create a JSON-comparible dict from a FitResults object

    Parameters:
        fit_results (FitResults): object containing fit parameters,\
            errors and type
        min_bound: optional min value to add to dictionary if min isn't\
            a fit parameter.
        max_bound: optional max value to add to dictionary if max isn't\
            a fit parameter.

    Returns:
        JSON-compatible dictionary with fit results

    Note:
        Supported fit types: 'norm', 'expon', 'uniform'
    '''

    type_map = {'norm': 'normal', 'expon': 'exponential', 'uniform': 'uniform'}
    param_map = {'uniform': lambda p: [('min', p[0]), ('max', p[0] + p[1])],
                 'norm': lambda p: [('mu', p[0]), ('sigma', p[1])],
                 'expon': lambda p: [('lambda', 1.0 / p[1])]}

    d = OrderedDict({'type': type_map[fit_results.type]})
    d.update(param_map[fit_results.type](fit_results.params))

    if min_bound is not None and 'min' not in d:
        d['min'] = min_bound
    if max_bound is not None and 'max' not in d:
        d['max'] = max_bound

    return d


def fit(data, distribution='norm'):
    '''Calculate the parameters of a fit of a distribution to a data set

    Parameters:
        data: array of data points to be fitted

    Options:
        distribution (str): type of distribution to fit. Default 'norm'.

    Returns:
        FitResults object with fitted parameters, errors and distrubution type

    Note:
        Uses Kolmogorov-Smirnov test to estimate distance and p-value.
    '''
    params = getattr(_st, distribution).fit(data)
    return FitResults(params, _st.kstest(data, distribution, params), distribution)


def optimal_distribution(data, distr_to_check=('norm', 'expon', 'uniform')):
    '''Calculate the parameters of a fit of different distributions to a data set
       and returns the distribution of the minimal ks-distance.

    Parameters:
        data: array of data points to be fitted

    Options:
        distr_to_check: tuple of distributions to be checked

    Returns:
        FitResults object with fitted parameters, errors and distrubution type\
            of the fit with the smallest fit distance

    Note:
        Uses Kolmogorov-Smirnov test to estimate distance and p-value.
    '''
    fit_results = [fit(data, d) for d in distr_to_check]
    return min(fit_results, key=lambda fit: fit.errs[0])


def scalar_stats(data, functions=('min', 'max', 'mean', 'std')):
    '''Calculate the stats from the given numpy functions

    Parameters:
        data: array of data points to be used for the stats

    Options:
        functions: tuple of numpy stat functions to apply on data

    Returns:
        Dictionary with tha name of the function as key and the result
        as the respective value
    '''
    stats = {}
    for func in functions:

        stats[func] = getattr(np, func)(data)

    return stats


def compare_two(data1, data2, test=StatTests.ks):
    '''Compares two distributions of data
       and assess two scores: a distance between them
       and a probability they are drawn from the same
       distribution.

    Parameters:
        data1: numpy array of dataset 1
        data2: numpy array of dataset 2
        test: Stat_tests\
            Defines the statistical test to be used, based\
            on the scipy available modules.\
            Accepted tests: ks_2samp, wilcoxon, ttest

    Returns:
        dist: float\
            High numbers define high dissimilarity between the two datasets
        p-value: float\
            Small numbers define high probability the data come from\
            same dataset.
    '''
    results = getattr(_st, get_test(test))(data1, data2)
    Stats = namedtuple('Stats', ['dist', 'pvalue'])

    return Stats(*results)


def total_score(paired_dats, p=2, test=StatTests.ks):
    '''Calculates the p-norm of the distances that have been calculated from the statistical
    test that has been applied on all the paired datasets.

    Parameters:
        paired_dats: a list of tuples or where each tuple
                         contains the paired data lists from two datasets

    Options:
        p : integer that defines the order of p-norm
        test: Stat_tests\
            Defines the statistical test to be used, based\
            on the scipy available modules.\
            Accepted tests: ks_2samp, wilcoxon, ttest

    Returns:
        A float corresponding to the p-norm of the distances that have
        been calculated. 0 corresponds to high similarity while 1 to low.
    '''
    scores = np.array([compare_two(fL1, fL2, test=test).dist for fL1, fL2 in paired_dats])
    return np.linalg.norm(scores, p)
