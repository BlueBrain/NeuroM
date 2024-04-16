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
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
from morphio import SomaError

import neurom as nm
from neurom.apps import EXAMPLE_STATS_CONFIG, get_config
from neurom.core.morphology import Morphology, Neurite
from neurom.core.population import Population
from neurom.exceptions import ConfigError
from neurom.features import (
    _MORPHOLOGY_FEATURES,
    _NEURITE_FEATURES,
    _POPULATION_FEATURES,
    _get_feature_value_and_func,
)
from neurom.io.utils import get_files_by_path
from neurom.utils import NeuromJSON, flatten

L = logging.getLogger(__name__)

IGNORABLE_EXCEPTIONS = {'SomaError': SomaError}


def _run_extract_stats(morph, config, process_subtrees):
    """The function to be called by multiprocessing.Pool.imap_unordered."""
    if not isinstance(morph, (Morphology, Population)):
        morph = nm.load_morphologies(morph, process_subtrees=process_subtrees)
    return morph.name, extract_stats(morph, config)


def extract_dataframe(morphs, config, n_workers=1, process_subtrees=False):
    """Extract stats grouped by neurite type from morphs.

    Arguments:
        morphs: a morphology, population, neurite tree, list of populations or list of morphology
            paths
        config (dict): configuration dict. The keys are:
            - neurite_type: a list of neurite types for which features are extracted
              If not provided, all neurite_type will be used
            - neurite:
                Either a list of features: [feature_name, {kwargs: {}, modes: []}] or
                a dictionary of features {feature_name: {kwargs: {}, modes: []}}.
                - kwargs is an optional entry allowing to pass kwargs to the feature function
                - modes is an aggregation operation provided as a string such as:
                  ['min', 'max', 'median', 'mean', 'std', 'raw', 'sum']
            - morphology: same as neurite entry, but it will not be run on each neurite_type,
              but only once on the whole morphology.
        n_workers (int): number of workers for multiprocessing (on collection of morphs)

    Returns:
        The extracted statistics

    Note:
        An example config can be found at:

    """
    if isinstance(morphs, Morphology):
        morphs = [morphs]
    elif isinstance(morphs, Population):
        morphs = morphs._files  # pylint: disable=protected-access

    func = partial(_run_extract_stats, config=config, process_subtrees=process_subtrees)
    if n_workers == 1:
        stats = list(map(func, morphs))
    else:
        if any(isinstance(i, Morphology) for i in morphs):
            raise ValueError("Can only process morphologies given as file paths when n_workers > 1")
        if n_workers > os.cpu_count():
            warnings.warn(f'n_workers ({n_workers}) > os.cpu_count() ({os.cpu_count()}))')
        with multiprocessing.Pool(n_workers) as pool:
            stats = list(pool.imap(func, morphs))

    columns = [('property', 'name')] + [
        (key1, key2) for key1, data in stats[0][1].items() for key2 in data
    ]
    rows = [
        [name] + list(flatten(features.values() for features in data.values()))
        for name, data in stats
    ]
    return pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns), data=rows)


extract_dataframe.__doc__ = extract_dataframe.__doc__.strip() + "\n\t" + str(EXAMPLE_STATS_CONFIG)


def _get_feature_stats(feature_name, morphs, modes, **kwargs):
    """Insert the stat data in the dict.

    If the feature is 2-dimensional, the feature is flattened on its last axis
    """

    def stat_name_format(mode, feature_name, **kwargs):
        """Returns the key name for the data dictionary.

        The key is a combination of the mode, feature_name and an optional suffix of all the extra
        kwargs that are passed in the feature function (apart from neurite_type).
        """
        suffix = "__".join(
            [f"{key}:{value}" for key, value in kwargs.items() if key != "neurite_type"]
        )

        if suffix:
            return f"{mode}_{feature_name}__{suffix}"

        return f"{mode}_{feature_name}"

    data = {}
    value, func = _get_feature_value_and_func(feature_name, morphs, **kwargs)
    shape = func.shape
    if len(shape) > 2:
        raise ValueError(f'Len of "{feature_name}" feature shape must be <= 2')  # pragma: no cover

    for mode in modes:
        stat_name = stat_name_format(mode, feature_name, **kwargs)

        stat = value
        if isinstance(value, Sized):
            if len(value) == 0 and mode not in {'raw', 'sum'}:
                stat = None
            elif mode == 'raw':
                stat = value
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
            - neurite:
                Either a list of features: [feature_name, {kwargs: {}, modes: []}] or
                a dictionary of features {feature_name: {kwargs: {}, modes: []}}.
                - kwargs is an optional entry allowing to pass kwargs to the feature function
                - modes is an aggregation operation provided as a string such as:
                  ['min', 'max', 'median', 'mean', 'std', 'raw', 'sum']
                - neurite_feature is a string from NEURITEFEATURES or NEURONFEATURES
                - mode is an aggregation operation provided as a string such as:
                  ['min', 'max', 'median', 'mean', 'std', 'raw', 'sum']
            - morphology: same as neurite entry, but it will not be run on each neurite_type,
              but only once on the whole morphology.

    Returns:
        The extracted statistics

    Note:
        An example config can be found in the `CLI -> neurom stats` page of the documentation.

    """
    # pylint: disable=too-many-nested-blocks
    config = _sanitize_config(config)

    neurite_types = [_NEURITE_MAP[t] for t in config.get('neurite_type', _NEURITE_MAP.keys())]

    stats = defaultdict(dict)
    for category in ("neurite", "morphology", "population"):
        for feature_name, opts in config[category].items():
            list_of_kwargs = opts["kwargs"]
            modes = opts["modes"]

            for feature_kwargs in list_of_kwargs:
                if category == 'neurite':
                    # mutated below, need a copy
                    feature_kwargs = deepcopy(feature_kwargs)

                    types = (
                        neurite_types
                        if 'neurite_type' not in feature_kwargs and neurite_types
                        else [_NEURITE_MAP[feature_kwargs.get('neurite_type', 'ALL')]]
                    )

                    for neurite_type in types:
                        if not isinstance(morphs, Neurite):
                            feature_kwargs["neurite_type"] = neurite_type
                        stats[neurite_type.name].update(
                            _get_feature_stats(
                                feature_name,
                                morphs,
                                modes,
                                **feature_kwargs,
                            )
                        )

                else:
                    stats[category].update(
                        _get_feature_stats(feature_name, morphs, modes, **feature_kwargs)
                    )

    return dict(stats)


extract_stats.__doc__ = extract_stats.__doc__.strip() + "\n\t" + str(EXAMPLE_STATS_CONFIG)


def _get_header(results):
    """Extracts the headers, using the first value in the dict as the template."""
    values = next(iter(results.values()))

    return ['name'] + [f'{k}:{metric}' for k, v in values.items() for metric in v.keys()]


def _generate_flattened_dict(headers, results):
    """Extract from results the fields in the headers list."""
    for name, values in results.items():
        row = []
        for header in headers:
            if header == 'name':
                row.append(name)
            else:
                # split on first occurence of `:` because feature kwargs may
                # use a colon for separating key and value.
                neurite_type, metric = header.split(':', 1)
                row.append(values[neurite_type][metric])
        yield row


_NEURITE_MAP = {
    'AXON': nm.AXON,
    'BASAL_DENDRITE': nm.BASAL_DENDRITE,
    'APICAL_DENDRITE': nm.APICAL_DENDRITE,
    'ALL': nm.ANY_NEURITE,
}


def full_config():
    """Returns a config with all features, all modes, all neurite types."""
    modes = ['min', 'max', 'median', 'mean', 'std']

    categories = {
        "neurite": _NEURITE_FEATURES,
        "morphology": _MORPHOLOGY_FEATURES,
        "population": _POPULATION_FEATURES,
    }

    config = {
        category: {name: {"kwargs": [{}], "modes": modes} for name in features}
        for category, features in categories.items()
    }

    config["neurite_type"] = list(_NEURITE_MAP.keys())

    return config


def _standardize_layout(category_features):
    """Standardizes the dictionary of features to a single format.

    Args:
        category_features: A dictionary the keys of which are features names and its values are
        either a list of modes ([]), a dictionary of kwargs and modes {kwargs: {}, modes: []}, or
        the standardized layout where the kwargs take a list of dicts {kwargs: [{}], modes: []}.

    Returns:
        The standardized features layout {feature: {kwargs: [{}], modes: []}}

    Notes:
        And example of the final layout is:

            - feature1:
                kwargs:
                  - kwargs1
                  - kwargs2
                modes:
                  - mode1
                  - mode2
            - feature2:
                kwargs:
                  - kwargs1
                  - kwargs2
                modes:
                  - mode1
                  - mode2
    """

    def standardize_options(options):
        """Returns options as a dict with two keys: 'kwargs' and 'modes'."""
        # convert short format
        if isinstance(options, list):
            return {"kwargs": [{}], "modes": options}

        modes = options.get("modes", [])

        if "kwargs" not in options:
            return {"kwargs": [{}], "modes": modes}

        kwargs = options["kwargs"]

        # previous format where kwargs were a single entry
        if isinstance(kwargs, dict):
            return {"kwargs": [kwargs], "modes": modes}

        return {"kwargs": kwargs, "modes": modes}

    return {name: standardize_options(options) for name, options in category_features.items()}


def _sanitize_config(config):
    """Check that the config has the correct keys, add missing keys if necessary."""
    config = deepcopy(config)

    if "neuron" in config:
        config["morphology"] = config.pop("neuron")

    for category in ("neurite", "morphology", "population"):
        config[category] = _standardize_layout(config[category]) if category in config else {}

    return config


def main(
    datapath,
    config,
    output_file,
    is_full_config,
    as_population,
    ignored_exceptions,
    use_subtrees=False,
):
    """Main function that get statistics for morphologies.

    Args:
        datapath (str|Path): path to a morphology file or folder
        config (str|Path): path to a statistics config file
        output_file (str|Path): path to output the resulted statistics file
        is_full_config (bool): should be statistics made over all possible features, modes, neurites
        as_population (bool): treat ``datapath`` as directory of morphologies population
        ignored_exceptions (list|tuple|None): exceptions to ignore when loading a morphology
        use_subtrees (bool): Enable of heterogeneous subtree processing
    """
    config = full_config() if is_full_config else get_config(config, EXAMPLE_STATS_CONFIG)

    if 'neurite' in config and 'neurite_type' not in config:
        error = ConfigError('"neurite_type" missing from config, but "neurite" set')
        L.error(error)
        raise error

    if ignored_exceptions is None:
        ignored_exceptions = ()

    morphs = nm.load_morphologies(
        get_files_by_path(datapath),
        ignored_exceptions=tuple(IGNORABLE_EXCEPTIONS[k] for k in ignored_exceptions),
        process_subtrees=use_subtrees,
    )

    if as_population:
        results = {datapath: extract_stats(morphs, config)}
    else:
        results = {m.name: extract_stats(m, config) for m in morphs}

    if not output_file:
        print(json.dumps(results, indent=2, separators=(',', ':'), cls=NeuromJSON))
    elif output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(results, f, cls=NeuromJSON)
    else:
        with open(output_file, 'w') as f:
            csvwriter = csv.writer(f)
            header = _get_header(results)
            csvwriter.writerow(header)
            for line in _generate_flattened_dict(header, dict(results)):
                csvwriter.writerow(line)
