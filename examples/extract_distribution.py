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

'''Extract a distribution for the selected feature of the population of neurons among
   the exponential, normal and uniform distribution, according to the minimum ks distance.
   '''

from neurom import ezy
from neurom import stats as st
import argparse


def parse_args():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(
        description='Morphology fit distribution extractor',
        epilog='Note: Prints the optimal distribution and corresponding parameters.')

    parser.add_argument('datapath',
                        help='Path to morphology data file or directory')

    parser.add_argument('feature',
                        help='Feature available for the ezy.neuron')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    data_path = args.datapath

    feature = args.feature

    population = ezy.load_neurons(data_path)

    feature_data = [getattr(n, 'get_' + feature)() for n in population]

    try:
        result = st.optimal_distribution(feature_data)
    except ValueError:
        from itertools import chain
        feature_data = list(chain(*feature_data))
        result = st.optimal_distribution(feature_data)

    print "Optimal distribution fit for %s is: %s with parameters %s"\
        % (feature, result[0], result[1][0])
