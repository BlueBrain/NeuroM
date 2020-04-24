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

"""Extract the optimal distributions for the following features of the population of neurons:
   soma: radius
   basal dendrites: n_neurites
   apical dendrites: n_neurites
   axons: n_neurites
"""

import argparse
from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple
from itertools import chain
import os
import json
from json import encoder

from neurom import get, load_neurons, NeuriteType, stats
from neurom.io.utils import get_morph_files

encoder.FLOAT_REPR = lambda o: format(o, '.2f')

Limits = namedtuple('Limits', 'min, max')
Feature = namedtuple('Feature', 'name, limits')
Component = namedtuple('Component', 'name, features')


# Map feature names to data getter namer
FEATURE_MAP = {
    'radius': 'soma_radii',
    'number': 'number_of_neurites',
    'segment_length': 'segment_lengths',
    'initial_radius': 'trunk_origin_radii',
    # 'taper_rate': lambda n, kwargs: np.array(list(n.iter_segments(mm.segment_taper_rate,
    #                                                              **kwargs))),
}

# Map component name to filtering parameters where applicable
PARAM_MAP = {
    'basal_dendrite': {'neurite_type': NeuriteType.basal_dendrite},
    'apical_dendrite': {'neurite_type': NeuriteType.apical_dendrite},
    'axon': {'neurite_type': NeuriteType.axon},
    'soma': None
}


def extract_data(neurons, feature, params=None):
    """Extracts feature from a list of neurons
       and transforms the fitted distribution in the correct format.
       Returns the optimal distribution and corresponding parameters.
       Normal distribution params (mean, std)
       Exponential distribution params (loc, scale)
       Uniform distribution params (min, range)
    """
    if params is None:
        params = {}

    feature_data = [get(FEATURE_MAP[feature], n, **params) for n in neurons]
    feature_data = list(chain(*feature_data))
    return stats.optimal_distribution(feature_data)


def transform_header(mtype_name):
    """Add header to json output to wrap around distribution data.
    """
    head_dict = OrderedDict()

    head_dict["m-type"] = mtype_name
    head_dict["components"] = defaultdict(OrderedDict)

    return head_dict


def transform_package(mtype, files, components):
    """Put together header and list of data into one json output.
       feature_list contains all the information about the data to be extracted:
       features, feature_names, feature_components, feature_min, feature_max
    """
    data_dict = transform_header(mtype)
    neurons = load_neurons(files)

    for comp in components:
        params = PARAM_MAP[comp.name]
        for feature in comp.features:
            result = stats.fit_results_to_dict(
                extract_data(neurons, feature.name, params),
                feature.limits.min, feature.limits.max
            )

            # When the distribution is normal with sigma = 0
            # it will be replaced with constant
            if result['type'] == 'normal' and result['sigma'] == 0.0:
                replace_result = OrderedDict((('type', 'constant'),
                                              ('val', result['mu'])))
                result = replace_result

            data_dict["components"][comp.name].update({feature.name: result})

    return data_dict


def get_mtype_from_filename(filename, sep='_'):
    """Get mtype of a morphology file from file name

    Assumes file name has structure 'a/b/c/d/mtype_xyx.abc'
    """
    return os.path.basename(filename).split(sep)[0]


def get_mtype_from_directory(filename):
    """Get mtype of a morphology file from file's parent directory name

    Assumes file name has structure 'a/b/c/mtype/xyx.abc'
    """
    return os.path.split(os.path.dirname(filename))[-1]


MTYPE_GETTERS = {
    'filename': get_mtype_from_filename,
    'directory': get_mtype_from_directory
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Morphology fit distribution extractor',
        epilog='Note: Outputs json of the optimal distribution \
                and corresponding parameters.')

    parser.add_argument('datapaths',
                        nargs='+',
                        help='Morphology data directory paths')

    parser.add_argument('--mtype', choices=MTYPE_GETTERS.keys(),
                        help='Get mtype from filename or parent directory')

    return parser.parse_args()


# Note: new features require a data getter function in
# FEATURE_MAP
# TODO: Set this config from a JSON or YAML file
COMPONENTS = [
    Component('soma',
              [Feature('radius', Limits(None, None))]),
    Component('basal_dendrite',
              [
                  Feature('number', Limits(0, None)),
                  Feature('segment_length', Limits(0, None)),
                  Feature('initial_radius', Limits(0, None)),
                  # Feature('taper_rate', Limits(0, None))
              ]),
    Component('apical_dendrite',
              [
                  Feature('number', Limits(0, None)),
                  Feature('segment_length', Limits(0, None)),
                  Feature('initial_radius', Limits(0, None)),
                  # Feature('taper_rate', Limits(0, None))
              ]),
    Component('axon',
              [
                  Feature('number', Limits(0, None)),
                  Feature('segment_length', Limits(0, None)),
                  Feature('initial_radius', Limits(0, None)),
                  # Feature('taper_rate', Limits(0, None))
              ]),
]


if __name__ == '__main__':
    args = parse_args()

    data_dirs = args.datapaths

    mtype_getter = MTYPE_GETTERS.get(args.mtype, lambda f: 'UNKNOWN')

    for d in data_dirs:
        mtype_files = defaultdict(list)
        for f in get_morph_files(d):
            mtype_files[mtype_getter(f)].append(f)

        _results = [transform_package(mtype_, files_, COMPONENTS)
                    for mtype_, files_ in mtype_files.items()]

        for res in _results:
            print(json.dumps(res, indent=2), '\n')
