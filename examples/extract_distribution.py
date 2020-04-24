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

"""Extract a distribution for the selected feature of the population of neurons among
   the exponential, normal and uniform distribution, according to the minimum ks distance.
   """

from itertools import chain
import argparse
import json

import neurom as nm
from neurom import stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Morphology fit distribution extractor',
        epilog='Note: Outputs json of the optimal distribution \
                and corresponding parameters.')

    parser.add_argument('datapath',
                        help='Path to morphology data directory')

    parser.add_argument('feature',
                        help='Feature to be extracted with neurom.get')

    return parser.parse_args()


def extract_data(data_path, feature):
    """Loads a list of neurons, extracts feature
       and transforms the fitted distribution in the correct format.
       Returns the optimal distribution, corresponding parameters,
       minimun and maximum values.
    """
    population = nm.load_neurons(data_path)

    feature_data = [nm.get(feature, n) for n in population]
    feature_data = list(chain(*feature_data))

    return stats.optimal_distribution(feature_data)


if __name__ == '__main__':
    args = parse_args()

    d_path = args.datapath

    feat = args.feature

    _result = stats.fit_results_to_dict(extract_data(d_path, feat))

    print(json.dumps(_result, indent=2, separators=(',', ': ')))
