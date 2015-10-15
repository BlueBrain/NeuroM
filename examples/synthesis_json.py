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

'''Extract the optimal distributions for the following features of the population of neurons:
   soma: radius
   basal dendrites: n_neurites
   apical dendrites: n_neurites
   axons: n_neurites
   '''

from neurom import ezy
from neurom import stats
import numpy as np
import argparse
import json
from collections import OrderedDict
import os


def parse_args():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(
        description='Morphology fit distribution extractor',
        epilog='Note: Outputs json of the optimal distribution \
                and corresponding parameters.')

    parser.add_argument('datapath',
                        help='Path to morphology data file or directory')

    return parser.parse_args()


def extract_data(data_path, feature, params=None):
    '''Loads a list of neurons, extracts feature
       and transforms the fitted distribution in the correct format.
       Returns the optimal distribution, corresponding parameters,
       minimun and maximum values.
    '''
    population = ezy.load_neurons(data_path)

    if params is None:
        params = {}

    feature_data = [getattr(n, 'get_' + feature)(*params) for n in population]

    results = {}

    try:
        opt_fit = stats.optimal_distribution(feature_data)
    except ValueError:
        from itertools import chain
        feature_data = list(chain(*feature_data))
        opt_fit = stats.optimal_distribution(feature_data)

    results["type"] = opt_fit[0]
    results["params"] = opt_fit[1][0]

    return results


def transform_distribution(data, datamin=None, datamax=None):
    '''Transform optimal distribution data into correct format
       based on the distribution type.
    '''
    data_dict = OrderedDict()

    if data["type"] == 'norm':
        data_dict.update({"type": "normal"})
        data_dict.update({"mu": data["params"][0]})
        data_dict.update({"sigma": data["params"][1]})

    elif data["type"] == 'expon':
        data_dict.update({"type": "exponential"})
        data_dict.update({"lambda": 1. / data["params"][1]})

    elif data["type"] == 'uniform':
        data_dict.update({"type": "uniform"})

    if datamin is not None:
        data_dict.update({"min": datamin})
    if datamax is not None:
        data_dict.update({"max": datamax})

    return data_dict


def transform_header(mtype_name):
    '''Add header to json output to wrap around distribution data.
    '''
    head_dict = OrderedDict()

    head_dict.update({"m-type": mtype_name})
    head_dict.update({"components": {}})

    return head_dict


def transform_package(data_path, feature_list, feature_names, component):
    '''Put together header and list of data into one json output.
    '''
    data_dict = transform_header(os.path.basename(data_path))

    for comp in np.unique(component):

        data_dict["components"].setdefault(comp)

    for feature, name, comp in zip(feature_list, component, feature_names):

        result = transform_distribution(extract_data(data_path, feature))
        data_dict["components"][comp] = {name: result}

    return data_dict

if __name__ == '__main__':
    args = parse_args()

    d_path = args.datapath

    flist = ["soma_radius"]
    fnames = ["radius"]
    comps = ["soma"]

    _result = transform_package(d_path, flist, fnames, comps)

    print json.dumps(_result, indent=2, separators=(', \n', ': '))
