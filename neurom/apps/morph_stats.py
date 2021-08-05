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
import os
import warnings
from collections import defaultdict
from collections.abc import Sized
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
from neurom.core.morphology import Morphology
from neurom.exceptions import ConfigError
from neurom.features import _NEURITE_FEATURES, _MORPHOLOGY_FEATURES, _POPULATION_FEATURES, \
    _get_feature_value_and_func
from neurom.io.utils import get_files_by_path
from neurom.utils import NeuromJSON, warn_deprecated

L = logging.getLogger(__name__)

EXAMPLE_CONFIG = Path(pkg_resources.resource_filename('neurom.apps', 'config'), 'morph_stats.yaml')
IGNORABLE_EXCEPTIONS = {'SomaError': SomaError}


def _run_extract_stats(morph, config):
    """The function to be called by multiprocessing.Pool.imap_unordered."""
    if not isinstance(morph, Morphology):
        morph = nm.load_morphology(morph)
    return morph.name, extract_stats(morph, config)


def extract_dataframe(morphs, config, n_workers=1):
    """Extract stats grouped by neurite type from morphs.

    Arguments:
        morphs: a morphology, population, neurite tree or list of morphology paths
        config (dict): configuration dict. The keys are:
            - neurite_type: a list of neurite types for which features are extracted
              If not provided, all neurite_type will be used
            - neurite: a dictionary {{neurite_feature: mode}} where:
                - neurite_feature is a string from NEURITEFEATURES or NEURONFEATURES
                - mode is an aggregation operation provided as a string such as:
                  ['min', 'max', 'median', 'mean', 'std', 'raw', 'sum']
            - morphology: same as neurite entry, but it will not be run on each neurite_type,
              but only once on the whole morphology.
        n_workers (int): number of workers for multiprocessing (on collection of morphs)

    Returns:
        The extracted statistics

    Note:
        An example config can be found at:

    {config_path}
    """
    if isinstance(morphs, Morphology):
        morphs = [morphs]
    config = config.copy()

    func = partial(_run_extract_stats, config=config)
    if n_workers == 1:
        stats = list(map(func, morphs))
    else:
        if n_workers > os.cpu_count():
            warnings.warn(f'n_workers ({n_workers}) > os.cpu_count() ({os.cpu_count()}))')
        with multiprocessing.Pool(n_workers) as pool:
            stats = list(pool.imap(func, morphs))

    columns = [('property', 'name')] + [
        (key1, key2) for key1, data in stats[0][1].items() for key2 in data
    ]
    rows = [[name] + list(chain.from_iterable(features.values() for features in data.values()))
            for name, data in stats]
    return pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns), data=rows)


extract_dataframe.__doc__ = extract_dataframe.__doc__.format(config_path=EXAMPLE_CONFIG)


def _get_feature_stats(feature_name, morphs, modes, kwargs):
    """Insert the stat data in the dict.

    If the feature is 2-dimensional, the feature is flattened on its last axis
    """
    data = {}
    value, func = _get_feature_value_and_func(feature_name, morphs, **kwargs)
    shape = func.shape
    if len(shape) > 2:
        raise ValueError(f'Len of "{feature_name}" feature shape must be <= 2')  # pragma: no cover

    for mode in modes:
        stat_name = f'{mode}_{feature_name}'

        stat = value
        if isinstance(value, Sized):
            if len(value) == 0 and mode not in {'raw', 'sum'}:
                stat = None
            else:
                stat = getattr(np, mode)(value, axis=0)

        if len(shape) == 2:
            for i in range(shape[1]):
                data[f'{stat_name}_{i}'] = stat[i] if stat is not None else None
        else:
            data[stat_name] = stat
    return data


def extract_stats(morphs, config):
    """Extract stats from morphs.

    Arguments:
        morphs: a morphology, population, neurite tree or list of morphology paths/str
        config (dict): configuration dict. The keys are:
            - neurite_type: a list of neurite types for which features are extracted
              If not provided, all neurite_type will be used.
            - neurite: a dictionary {{neurite_feature: mode}} where:
                - neurite_feature is a string from NEURITEFEATURES or NEURONFEATURES
                - mode is an aggregation operation provided as a string such as:
                  ['min', 'max', 'median', 'mean', 'std', 'raw', 'sum']
            - morphology: same as neurite entry, but it will not be run on each neurite_type,
              but only once on the whole morphology.

    Returns:
        The extracted statistics

    Note:
        An example config can be found at:

    {config_path}
    """
    stats = defaultdict(dict)
    neurite_features = product(['neurite'], config.get('neurite', {}).items())
    if 'neuron' in config:    # pragma: no cover
        warn_deprecated('Usage of "neuron" is deprecated in configs of `morph_stats` package. '
                        'Use "morphology" instead.')
        config['morphology'] = config['neuron']
        del config['neuron']
    morph_features = product(['morphology'], config.get('morphology', {}).items())
    population_features = product(['population'], config.get('population', {}).items())
    neurite_types = [_NEURITE_MAP[t] for t in config.get('neurite_type', _NEURITE_MAP.keys())]

    for namespace, (feature_name, opts) in chain(neurite_features, morph_features,
                                                 population_features):
        if isinstance(opts, dict):
            kwargs = opts.get('kwargs', {})
            modes = opts.get('modes', [])
        else:
            kwargs = {}
            modes = opts
        if namespace == 'neurite':
            if 'neurite_type' not in kwargs and neurite_types:
                for t in neurite_types:
                    kwargs['neurite_type'] = t
                    stats[t.name].update(_get_feature_stats(feature_name, morphs, modes, kwargs))
            else:
                t = _NEURITE_MAP[kwargs.get('neurite_type', 'ALL')]
                kwargs['neurite_type'] = t
                stats[t.name].update(_get_feature_stats(feature_name, morphs, modes, kwargs))
        else:
            stats[namespace].update(_get_feature_stats(feature_name, morphs, modes, kwargs))

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
        'neurite': {feature: modes for feature in _NEURITE_FEATURES},
        'morphology': {feature: modes for feature in _MORPHOLOGY_FEATURES},
        'population': {feature: modes for feature in _POPULATION_FEATURES},
        'neurite_type': list(_NEURITE_MAP.keys()),
    }


def sanitize_config(config):
    """Check that the config has the correct keys, add missing keys if necessary."""
    if 'neurite' in config:
        if 'neurite_type' not in config:
            raise ConfigError('"neurite_type" missing from config, but "neurite" set')
    else:
        config['neurite'] = {}

    if 'morphology' not in config:
        config['morphology'] = {}

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
    morphs = nm.load_morphologies(get_files_by_path(datapath),
                                  ignored_exceptions=ignored_exceptions)

    results = {}
    if as_population:
        results[datapath] = extract_stats(morphs, config)
    else:
        for m in tqdm(morphs):
            results[m.name] = extract_stats(m, config)

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
