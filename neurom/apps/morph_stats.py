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

"""Core code for morph_stats application."""
import logging
from collections import defaultdict
from itertools import product
import os
import pkg_resources


import numpy as np
import pandas as pd

import neurom as nm
from neurom.features import NEURONFEATURES, NEURITEFEATURES
from neurom.exceptions import ConfigError

L = logging.getLogger(__name__)

EXAMPLE_CONFIG = os.path.join(pkg_resources.resource_filename(
    'neurom', 'config'), 'morph_stats.yaml')


def eval_stats(values, mode):
    """Extract a summary statistic from an array of list of values.

    Arguments:
        values: A numpy array of values
        mode: A summary stat to extract. One of:
            ['min', 'max', 'median', 'mean', 'std', 'raw', 'total']

    Note: fails silently if values is empty, and None is returned
    """
    if mode == 'raw':
        return values.tolist()
    if mode == 'total':
        mode = 'sum'

    try:
        return getattr(np, mode)(values, axis=0)
    except ValueError:
        pass

    return None


def _stat_name(feat_name, stat_mode):
    """Set stat name based on feature name and stat mode."""
    if feat_name[-1] == 's':
        feat_name = feat_name[:-1]
    if feat_name == 'soma_radii':
        feat_name = 'soma_radius'
    if stat_mode == 'raw':
        return feat_name

    return '%s_%s' % (stat_mode, feat_name)


def _fill_compoundified(data, stat_name, stat):
    """Insert the stat in the dict and eventually split it into XYZ components."""
    if stat is None or not isinstance(stat, np.ndarray) or stat.shape not in ((3, ), ):
        data[stat_name] = stat
    else:
        for i, suffix in enumerate('XYZ'):
            compound_stat_name = stat_name + '_' + suffix
            data[compound_stat_name] = stat[i]


def extract_stats(neurons, config, as_frame=False):
    """Extract stats from neurons.

    Arguments:
        neurons: a neuron, population or neurite tree
        config (dict): configuration dict. The keys are:
            - neurite_type: a list of neurite types for which features are extracted
            - neurite: a dictionary {neurite_feature: mode} where:
                - neurite_feature is a string from NEURITEFEATURES
                - mode is an aggregation operation provided as a string such as:
                  ['min', 'max', 'median', 'mean', 'std', 'raw']
        as_frame (bool): if yes, returns a dataframe, else a dict

    Returns:
        The extracted statistics

    Note:
        An example config can be found at:
    """
    if as_frame:
        return _extract_stats_as_frame(neurons, config)
    return _extract_stats_as_dict(neurons, config)


extract_stats.__doc__ = extract_stats.__doc__ + EXAMPLE_CONFIG + '\n'


def _extract_stats_as_frame(neurons, config):
    del config['neuron']
    stats = {nrn.name: extract_stats(nrn, config) for nrn in neurons}
    columns = list(next(iter(next(iter(stats.values())).values())).keys())

    rows = [[name, neurite_type] + list(features.values())
            for name, data in stats.items()
            for neurite_type, features in data.items()]
    return pd.DataFrame(columns=['name', 'neurite_type'] + columns,
                        data=rows)


def _extract_stats_as_dict(neurons, config):
    """Extract stats from neurons."""
    stats = defaultdict(dict)

    for (feature_name, modes), neurite_type in product(config['neurite'].items(),
                                                       config['neurite_type']):
        neurite_type = _NEURITE_MAP[neurite_type]
        for mode in modes:
            stat_name = _stat_name(feature_name, mode)
            stat = eval_stats(nm.get(feature_name, neurons, neurite_type=neurite_type), mode)
            _fill_compoundified(stats[neurite_type.name], stat_name, stat)

    for feature_name, modes in config.get('neuron', {}).items():
        for mode in modes:
            stat_name = _stat_name(feature_name, mode)
            stat = eval_stats(nm.get(feature_name, neurons), mode)
            _fill_compoundified(stats, stat_name, stat)

    return dict(stats)


def get_header(results):
    """Extracts the headers, using the first value in the dict as the template."""
    ret = ['name', ]
    values = next(iter(results.values()))
    for k, v in values.items():
        if isinstance(v, dict):
            for metric in v.keys():
                ret.append('%s:%s' % (k, metric))
        else:
            ret.append(k)
    return ret


def generate_flattened_dict(headers, results):
    """Extract from results the fields in the headers list."""
    for name, values in results.items():
        row = []
        for header in headers:
            if header == 'name':
                row.append(name)
            elif ':' in header:
                neurite_type, metric = header.split(':')
                row.append(values[neurite_type][metric])
            else:
                row.append(values[header])
        yield row


_NEURITE_MAP = {
    'AXON': nm.AXON,
    'BASAL_DENDRITE': nm.BASAL_DENDRITE,
    'APICAL_DENDRITE': nm.APICAL_DENDRITE,
    'ALL': nm.ANY_NEURITE
}


def full_config():
    """Returns a config with all features, all modes, all neurite types."""
    modes = ['min', 'max', 'median', 'mean', 'std']
    return {
        'neurite': {feature: modes for feature in NEURITEFEATURES},
        'neuron': {feature: modes for feature in NEURONFEATURES},
        'neurite_type': list(_NEURITE_MAP.keys()),
    }


def sanitize_config(config):
    """Check that the config has the correct keys, add missing keys if necessary."""
    if 'neurite' in config:
        if 'neurite_type' not in config:
            raise ConfigError('"neurite_type" missing from config, but "neurite" set')
    else:
        config['neurite'] = {}

    if 'neuron' not in config:
        config['neuron'] = {}

    return config
