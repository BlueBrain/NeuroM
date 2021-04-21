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

"""Statistics for morphologies."""

import csv
import json
import logging
import multiprocessing
import numbers
import os
import warnings
from collections import defaultdict
from functools import partial
from itertools import chain, product
from pathlib import Path
import pkg_resources
from tqdm import tqdm
import numpy as np
import pandas as pd
from morphio import SomaError

import neurom as nm
from neurom.apps import get_config
from neurom.core._neuron import Neuron
from neurom.exceptions import ConfigError
from neurom.features import NEURITEFEATURES, NEURONFEATURES, _get_feature_value_and_func
from neurom.io.utils import get_files_by_path
from neurom.utils import NeuromJSON

L = logging.getLogger(__name__)

EXAMPLE_CONFIG = Path(pkg_resources.resource_filename('neurom.apps', 'config'), 'morph_stats.yaml')
IGNORABLE_EXCEPTIONS = {'SomaError': SomaError}


def eval_stats(values, mode):
    """Extract a summary statistic from an array of list of values.

    Arguments:
        values: A numpy array of values
        mode: A summary stat to extract. One of:
            ['min', 'max', 'median', 'mean', 'std', 'raw', 'total']

    .. note:: If values is empty, mode `raw` returns `[]`, `total` returns `0.0`
    and the other modes return `None`.
    """
    if mode == 'raw':
        return values.tolist()
    if mode == 'total':
        mode = 'sum'
    if len(values) == 0 and mode not in {'raw', 'sum'}:
        return None

    return getattr(np, mode)(values, axis=0)


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
    if not isinstance(nrn, Neuron):
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
                - neurite_feature is a string from NEURITEFEATURES or NEURONFEATURES
                - mode is an aggregation operation provided as a string such as:
                  ['min', 'max', 'median', 'mean', 'std', 'raw', 'total']
            - neuron: same as neurite entry, but it will not be run on each neurite_type,
              but only once on the whole neuron.
        n_workers (int): number of workers for multiprocessing (on collection of neurons)

    Returns:
        The extracted statistics

    Note:
        An example config can be found at:

    {config_path}
    """
    if isinstance(neurons, Neuron):
        neurons = [neurons]
    config = config.copy()

    func = partial(_run_extract_stats, config=config)
    if n_workers == 1:
        stats = list(map(func, neurons))
    else:
        if n_workers > os.cpu_count():
            warnings.warn(f'n_workers ({n_workers}) > os.cpu_count() ({os.cpu_count()}))')
        with multiprocessing.Pool(n_workers) as pool:
            stats = list(pool.imap(func, neurons))

    columns = [('property', 'name')] + [
        (key1, key2) for key1, data in stats[0][1].items() for key2 in data
    ]
    rows = [[name] + list(chain.from_iterable(features.values() for features in data.values()))
            for name, data in stats]
    return pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns), data=rows)


extract_dataframe.__doc__ = extract_dataframe.__doc__.format(config_path=EXAMPLE_CONFIG)


def extract_stats(neurons, config):
    """Extract stats from neurons.

    Arguments:
        neurons: a neuron, population, neurite tree or list of neuron paths/str
        config (dict): configuration dict. The keys are:
            - neurite_type: a list of neurite types for which features are extracted
              If not provided, all neurite_type will be used.
            - neurite: a dictionary {{neurite_feature: mode}} where:
                - neurite_feature is a string from NEURITEFEATURES or NEURONFEATURES
                - mode is an aggregation operation provided as a string such as:
                  ['min', 'max', 'median', 'mean', 'std', 'raw', 'total']
            - neuron: same as neurite entry, but it will not be run on each neurite_type,
              but only once on the whole neuron.

    Returns:
        The extracted statistics

    Note:
        An example config can be found at:

    {config_path}
    """

    def _fill_stats_dict(data, stat_name, stat, shape):
        """Insert the stat data in the dict.

        If the feature is 2-dimensional, the feature is flattened on its last axis
        """
        if len(shape) == 2:
            for i in range(shape[1]):
                data[f'{stat_name}_{i}'] = stat[i] if stat is not None else None
        elif len(shape) > 2:
            raise ValueError(f'Feature with wrong shape: {shape}')  # pragma: no cover
        else:
            data[stat_name] = stat

    stats = defaultdict(dict)

    for (feature_name, modes), neurite_type in product(config['neurite'].items(),
                                                       config.get('neurite_type',
                                                                  _NEURITE_MAP.keys())):
        neurite_type = _NEURITE_MAP[neurite_type]
        feature, func = _get_feature_value_and_func(feature_name, neurons,
                                                    neurite_type=neurite_type)
        if isinstance(feature, numbers.Number):
            feature = [feature]
        for mode in modes:
            stat_name = _stat_name(feature_name, mode)
            stat = eval_stats(feature, mode)
            _fill_stats_dict(stats[neurite_type.name], stat_name, stat, func.shape)

    for feature_name, modes in config.get('neuron', {}).items():
        feature, func = _get_feature_value_and_func(feature_name, neurons)
        if isinstance(feature, numbers.Number):
            feature = [feature]
        for mode in modes:
            stat_name = _stat_name(feature_name, mode)
            stat = eval_stats(feature, mode)
            _fill_stats_dict(stats['neuron'], stat_name, stat, func.shape)
    return dict(stats)


extract_stats.__doc__ = extract_stats.__doc__.format(config_path=EXAMPLE_CONFIG)


def get_header(results):
    """Extracts the headers, using the first value in the dict as the template."""
    ret = ['name', ]
    values = next(iter(results.values()))
    for k, v in values.items():
        for metric in v.keys():
            ret.append('%s:%s' % (k, metric))
    return ret


def generate_flattened_dict(headers, results):
    """Extract from results the fields in the headers list."""
    for name, values in results.items():
        row = []
        for header in headers:
            if header == 'name':
                row.append(name)
            else:
                neurite_type, metric = header.split(':')
                row.append(values[neurite_type][metric])
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


def main(datapath, config, output_file, is_full_config, as_population, ignored_exceptions):
    """Main function that get statistics for morphologies.

    Args:
        datapath (str|Path): path to a morphology file or folder
        config (str|Path): path to a statistics config file
        output_file (str|Path): path to output the resulted statistics file
        is_full_config (bool): should be statistics made over all possible features, modes, neurites
        as_population (bool): treat ``datapath`` as directory of morphologies population
        ignored_exceptions (list|tuple|None): exceptions to ignore when loading a morphology
    """
    if is_full_config:
        config = full_config()
    else:
        try:
            config = get_config(config, EXAMPLE_CONFIG)
            config = sanitize_config(config)
        except ConfigError as e:
            L.error(e)
            raise

    if ignored_exceptions is None:
        ignored_exceptions = ()
    ignored_exceptions = tuple(IGNORABLE_EXCEPTIONS[k] for k in ignored_exceptions)
    neurons = nm.load_neurons(get_files_by_path(datapath), ignored_exceptions=ignored_exceptions)

    results = {}
    if as_population:
        results[datapath] = extract_stats(neurons, config)
    else:
        for neuron in tqdm(neurons):
            results[neuron.name] = extract_stats(neuron, config)

    if not output_file:
        print(json.dumps(results, indent=2, separators=(',', ':'), cls=NeuromJSON))
    elif output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(results, f, cls=NeuromJSON)
    else:
        with open(output_file, 'w') as f:
            csvwriter = csv.writer(f)
            header = get_header(results)
            csvwriter.writerow(header)
            for line in generate_flattened_dict(header, dict(results)):
                csvwriter.writerow(line)
