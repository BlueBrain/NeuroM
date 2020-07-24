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
import os
import logging
from collections import defaultdict
from itertools import product
from pathlib import Path
import multiprocessing
from functools import partial
import warnings

import numpy as np
import pandas as pd
import pkg_resources

import neurom as nm
from neurom.exceptions import ConfigError
from neurom.features import NEURITEFEATURES, NEURONFEATURES
from neurom.fst._core import FstNeuron

L = logging.getLogger(__name__)

EXAMPLE_CONFIG = Path(pkg_resources.resource_filename(
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


def _run_extract_stats(nrn, config):
    """The function to be called by multiprocessing.Pool.imap_unordered."""
    if not isinstance(nrn, FstNeuron):
        nrn = nm.load_neuron(nrn)
    return nrn.name, extract_stats(nrn, config)


def extract_dataframe(neurons, config, n_workers=1):
    """Extract stats grouped by neurite type from neurons.

    Arguments:
        neurons: a neuron, population, neurite tree or list of neuron paths
        config (dict): configuration dict. The keys are:
            - neurite_type: a list of neurite types for which features are extracted
              If not provided, all neurite_type will be used
            - neurite: a dictionary {{neurite_feature: mode}} where:
                - neurite_feature is a string from NEURITEFEATURES
                - mode is an aggregation operation provided as a string such as:
                  ['min', 'max', 'median', 'mean', 'std', 'raw', 'total']
        n_workers (int): number of workers for multiprocessing (on collection of neurons)

    Returns:
        The extracted statistics

    Note:
        An example config can be found at:

    {config_path}
    """
    if isinstance(neurons, FstNeuron):
        neurons = [neurons]
    config = config.copy()

    # Only NEURITEFEATURES are considered since the dataframe is built by neurite_type
    # NEURONFEATURES are discarded
    if 'neuron' in config:
        del config['neuron']

    func = partial(_run_extract_stats, config=config)
    if n_workers == 1:
        stats = dict(map(func, neurons))
    else:
        if n_workers > os.cpu_count():
            warnings.warn(f'n_workers ({n_workers}) > os.cpu_count() ({os.cpu_count()}))')
        with multiprocessing.Pool(n_workers) as pool:
            stats = dict(pool.imap_unordered(func, neurons))

    columns = list(next(iter(next(iter(stats.values())).values())).keys())

    rows = [[name, neurite_type] + list(features.values())
            for name, data in stats.items()
            for neurite_type, features in data.items()]
    return pd.DataFrame(columns=['name', 'neurite_type'] + columns,
                        data=rows)


def extract_stats(neurons, config):
    """Extract stats from neurons.

    Arguments:
        neurons: a neuron, population, neurite tree or list of neuron paths/str
        config (dict): configuration dict. The keys are:
            - neurite_type: a list of neurite types for which features are extracted
              If not provided, all neurite_type will be used
            - neurite: a dictionary {{neurite_feature: mode}} where:
                - neurite_feature is a string from NEURITEFEATURES
                - mode is an aggregation operation provided as a string such as:
                  ['min', 'max', 'median', 'mean', 'std', 'raw', 'total']

    Returns:
        The extracted statistics

    Note:
        An example config can be found at:

    {config_path}
    """

    def _fill_stats_dict(data, stat_name, stat):
        """Insert the stat data in the dict.

        And if the stats is a 3D array, splits it into XYZ components.
        """
        if stat is None or not isinstance(stat, np.ndarray) or stat.shape not in ((3, ), ):
            data[stat_name] = stat
        else:
            for i, suffix in enumerate('XYZ'):
                compound_stat_name = stat_name + '_' + suffix
                data[compound_stat_name] = stat[i]

    stats = defaultdict(dict)

    for (feature_name, modes), neurite_type in product(config['neurite'].items(),
                                                       config.get('neurite_type',
                                                                  _NEURITE_MAP.keys())):
        neurite_type = _NEURITE_MAP[neurite_type]
        feature = nm.get(feature_name, neurons, neurite_type=neurite_type)
        for mode in modes:
            stat_name = _stat_name(feature_name, mode)
            stat = eval_stats(feature, mode)
            _fill_stats_dict(stats[neurite_type.name], stat_name, stat)

    for feature_name, modes in config.get('neuron', {}).items():
        feature = nm.get(feature_name, neurons)
        for mode in modes:
            stat_name = _stat_name(feature_name, mode)
            stat = eval_stats(feature, mode)
            _fill_stats_dict(stats, stat_name, stat)

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


extract_stats.__doc__ = extract_stats.__doc__.format(config_path=EXAMPLE_CONFIG)
extract_dataframe.__doc__ = extract_dataframe.__doc__.format(config_path=EXAMPLE_CONFIG)
