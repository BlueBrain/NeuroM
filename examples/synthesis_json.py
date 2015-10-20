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
   '''

from neurom import ezy
from neurom import stats
from neurom.io.utils import get_morph_files
import numpy as np
import argparse
import json
from collections import OrderedDict
from collections import defaultdict
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


def extract_data(files, feature, params=None):
    '''Loads a list of neurons, extracts feature
       and transforms the fitted distribution in the correct format.
       Returns the optimal distribution and corresponding parameters.
       Normal distribution params (mean, std)
       Exponential distribution params (loc, scale)
       Uniform distribution params (min, range)
    '''
    population = ezy.load_neurons(files)

    if params is None:
        params = {}

    feature_data = [getattr(n, 'get_' + feature)(**params) for n in population]

    try:
        opt_fit = stats.optimal_distribution(feature_data)
    except ValueError:
        from itertools import chain
        feature_data = list(chain(*feature_data))
        opt_fit = stats.optimal_distribution(feature_data)

    return opt_fit


def transform_header(mtype_name, components):
    '''Add header to json output to wrap around distribution data.
    '''
    head_dict = OrderedDict()

    head_dict["m-type"] = mtype_name
    head_dict["components"] = {}

    for comp in np.unique(components):
        head_dict["components"].setdefault(comp)

    return head_dict


def transform_package(mtype, files, components, feature_list):
    '''Put together header and list of data into one json output.
       feature_list contains all the information about the data to be extracted:
       features, feature_names, feature_components, feature_min, feature_max
    '''
    data_dict = transform_header(mtype, components)

    for feature, name, comp, fmin, fmax, fparam in feature_list:

        result = stats.fit_results_to_dict(extract_data(files, feature, fparam),
                                           fmin, fmax)

        data_dict["components"][comp] = {name: result}

    return data_dict


def get_mtype(filename, sep='_'):
    '''Get mtype of a morphology file from file name

    Assumes file name has structure 'a/b/c/d/mtype_xyx.abc'
    '''
    return os.path.basename(filename).split(sep)[0]

if __name__ == '__main__':
    args = parse_args()

    d_path = args.datapath

    mtype_files = defaultdict(list)

    for f in get_morph_files(d_path):
        mtype_files[get_mtype(f)].append(f)

    flist = [["soma_radius", "radius", "soma", None, None, None],
             ["n_neurites", "number", "basal_dendrite", 1, None,
              {"neurite_type": ezy.TreeType.basal_dendrite}]]

    comps = ["soma"]

    _results = [transform_package(mtype_, files_, comps, flist)
                for mtype_, files_ in mtype_files.iteritems()]

    for res in _results:
        print json.dumps(res, indent=2, separators=(', \n', ': ')), '\n'
