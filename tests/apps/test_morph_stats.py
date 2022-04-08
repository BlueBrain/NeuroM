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

import os
import warnings
from copy import deepcopy
from pathlib import Path

import neurom as nm
import pandas as pd
from neurom.apps import morph_stats as ms
from neurom.exceptions import ConfigError
from neurom.features import _NEURITE_FEATURES, _MORPHOLOGY_FEATURES, _POPULATION_FEATURES

import pytest
from numpy.testing import assert_array_equal, assert_almost_equal
from pandas.testing import assert_frame_equal

DATA_PATH = Path(__file__).parent.parent / 'data'
SWC_PATH = DATA_PATH / 'swc'
REF_CONFIG = {
    'neurite': {
        'section_lengths': ['max', 'sum'],
        'section_volumes': ['sum'],
        'section_branch_orders': ['max', 'raw'],
        'segment_midpoints': ['max'],
        'max_radial_distance': ['mean'],
    },
    'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL'],
    'morphology': {
        'soma_radius': ['mean'],
        'max_radial_distance': ['mean'],
    }
}

REF_CONFIG_NEW = {
    'neurite': {
        'section_lengths': {'modes': ['max', 'sum']},
        'section_volumes': {'modes': ['sum']},
        'section_branch_orders': {'modes': ['max', 'raw']},
        'segment_midpoints': {'modes': ['max']},
        'max_radial_distance': {'modes': ['mean']},
    },
    'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL'],
    'morphology': {
        'soma_radius': {'modes': ['mean']},
        'max_radial_distance': {'modes': ['mean']},
    }
}

REF_OUT = {
    'morphology': {
        'mean_soma_radius': 0.13065629648763766,
        'mean_max_radial_distance': 99.5894610648815,
    },
    'axon': {
        'sum_section_lengths': 207.87975220908129,
        'max_section_lengths': 11.018460736176685,
        'max_section_branch_orders': 10,
        'raw_section_branch_orders': [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        'sum_section_volumes': 276.73857657289523,
        'max_segment_midpoints_0': 0.0,
        'max_segment_midpoints_1': 0.0,
        'max_segment_midpoints_2': 49.520305964149998,
        'mean_max_radial_distance': 82.44254511788921,
    },
    'all': {
        'sum_section_lengths': 840.68521442251949,
        'max_section_lengths': 11.758281556059444,
        'max_section_branch_orders': 10,
        'raw_section_branch_orders': [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        'sum_section_volumes': 1104.9077419665782,
        'max_segment_midpoints_0': 64.401674984050004,
        'max_segment_midpoints_1': 48.48197694465,
        'max_segment_midpoints_2': 53.750947521650005,
        'mean_max_radial_distance': 99.5894610648815,
    },
    'apical_dendrite': {
        'sum_section_lengths': 214.37304577550353,
        'max_section_lengths': 11.758281556059444,
        'max_section_branch_orders': 10,
        'raw_section_branch_orders': [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        'sum_section_volumes': 271.9412385728449,
        'max_segment_midpoints_0': 64.401674984050004,
        'max_segment_midpoints_1': 0.0,
        'max_segment_midpoints_2': 53.750947521650005,
        'mean_max_radial_distance': 99.5894610648815,
    },
    'basal_dendrite': {
        'sum_section_lengths': 418.43241643793476,
        'max_section_lengths': 11.652508126101711,
        'max_section_branch_orders': 10,
        'raw_section_branch_orders': [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        'sum_section_volumes': 556.22792682083821,
        'max_segment_midpoints_0': 64.007872333250006,
        'max_segment_midpoints_1': 48.48197694465,
        'max_segment_midpoints_2': 51.575580778049996,
        'mean_max_radial_distance': 94.43342438865741,
    },
}


def test_extract_stats_single_morphology():
    m = nm.load_morphology(SWC_PATH / 'Neuron.swc')
    res = ms.extract_stats(m, REF_CONFIG)
    assert set(res.keys()) == set(REF_OUT.keys())
    for k in ('morphology', 'all', 'axon', 'basal_dendrite', 'apical_dendrite'):
        assert set(res[k].keys()) == set(REF_OUT[k].keys())
        for kk in res[k].keys():
            assert_almost_equal(res[k][kk], REF_OUT[k][kk], decimal=4)


def test_extract_stats_new_format():
    m = nm.load_morphology(SWC_PATH / 'Neuron.swc')
    res = ms.extract_stats(m, REF_CONFIG_NEW)
    assert set(res.keys()) == set(REF_OUT.keys())
    for k in ('morphology', 'all', 'axon', 'basal_dendrite', 'apical_dendrite'):
        assert set(res[k].keys()) == set(REF_OUT[k].keys())
        for kk in res[k].keys():
            assert_almost_equal(res[k][kk], REF_OUT[k][kk], decimal=4)


def test_stats_new_format_set_arg():
    m = nm.load_morphology(SWC_PATH / 'Neuron.swc')
    config = {
        'neurite': {
            'section_lengths': {'kwargs': {'neurite_type': 'AXON'}, 'modes': ['max', 'sum']},
        },
        'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL'],
        'morphology': {
            'soma_radius': {'modes': ['mean']},
        }
    }
    initial_config = deepcopy(config)

    res = ms.extract_stats(m, config)
    assert config == initial_config
    assert set(res.keys()) == {'morphology', 'axon'}
    assert set(res['axon'].keys()) == {'max_section_lengths', 'sum_section_lengths'}
    assert set(res['morphology'].keys()) == {'mean_soma_radius'}


def test_extract_stats_scalar_feature():
    m = nm.load_morphology(DATA_PATH / 'neurolucida' / 'bio_neuron-000.asc')
    config = {
        'neurite_type': ['ALL'],
        'neurite': {
            'number_of_forking_points': ['max'],
        },
        'morphology': {
            'soma_volume': ['sum'],
        }
    }
    res = ms.extract_stats(m, config)
    assert res == {'all': {'max_number_of_forking_points': 277},
                   'morphology': {'sum_soma_volume': 1424.4383771584492}}


def test_extract_dataframe():
    # Vanilla test
    initial_config = deepcopy(REF_CONFIG_NEW)

    morphs = nm.load_morphologies([SWC_PATH / 'Neuron.swc', SWC_PATH / 'simple.swc'])
    actual = ms.extract_dataframe(morphs, REF_CONFIG_NEW)

    # drop raw features as they require too much test data to mock
    actual = actual.drop(columns='raw_section_branch_orders', level=1)
    expected = pd.read_csv(Path(DATA_PATH, 'extracted-stats.csv'), header=[0, 1], index_col=0)
    assert_frame_equal(actual, expected, check_dtype=False)
    assert REF_CONFIG_NEW == initial_config

    # Test with a single morphology in the population
    morphs = nm.load_morphologies(SWC_PATH / 'Neuron.swc')
    actual = ms.extract_dataframe(morphs, REF_CONFIG_NEW)
    # drop raw features as they require too much test data to mock
    actual = actual.drop(columns='raw_section_branch_orders', level=1)
    assert_frame_equal(actual, expected.iloc[[0]], check_dtype=False)
    assert REF_CONFIG_NEW == initial_config

    # Test with a config without the 'morphology' key
    morphs = nm.load_morphologies([Path(SWC_PATH, name)
                                 for name in ['Neuron.swc', 'simple.swc']])
    config = {'neurite': {'section_lengths': ['sum']},
              'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL']}
    initial_config = deepcopy(config)
    actual = ms.extract_dataframe(morphs, config)
    idx = pd.IndexSlice
    expected = expected.loc[:, idx[:, ['name', 'sum_section_lengths']]]
    assert_frame_equal(actual, expected, check_dtype=False)
    assert config == initial_config

    # Test with a Morphology argument
    m = nm.load_morphology(Path(SWC_PATH, 'Neuron.swc'))
    actual = ms.extract_dataframe(m, config)
    assert_frame_equal(actual, expected.iloc[[0]], check_dtype=False)
    assert config == initial_config

    # Test with a List[Morphology] argument
    morphs = [nm.load_morphology(Path(SWC_PATH, name))
            for name in ['Neuron.swc', 'simple.swc']]
    actual = ms.extract_dataframe(morphs, config)
    assert_frame_equal(actual, expected, check_dtype=False)
    assert config == initial_config

    # Test with a List[Path] argument
    morphs = [Path(SWC_PATH, name) for name in ['Neuron.swc', 'simple.swc']]
    actual = ms.extract_dataframe(morphs, config)
    assert_frame_equal(actual, expected, check_dtype=False)
    assert config == initial_config

    # Test without any neurite_type keys, it should pick the defaults
    config = {'neurite': {'total_length_per_neurite': ['sum']}}
    initial_config = deepcopy(config)
    actual = ms.extract_dataframe(morphs, config)
    expected_columns = pd.MultiIndex.from_tuples(
        [('property', 'name'),
         ('axon', 'sum_total_length_per_neurite'),
         ('basal_dendrite', 'sum_total_length_per_neurite'),
         ('apical_dendrite', 'sum_total_length_per_neurite'),
         ('all', 'sum_total_length_per_neurite')])
    expected = pd.DataFrame(
        columns=expected_columns,
        data=[['Neuron.swc', 207.87975221, 418.43241644, 214.37304578, 840.68521442],
              ['simple.swc', 15.,          16.,           0.,          31., ]])
    assert_frame_equal(actual, expected, check_dtype=False)
    assert config == initial_config


def test_extract_dataframe_with_kwargs():
    config = {
        'neurite': {
            'section_lengths': {'kwargs': {'neurite_type': 'AXON'}, 'modes': ['max', 'sum']},
        },
        'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL'],
        'morphology': {
            'soma_radius': {'modes': ['mean']},
        }
    }
    initial_config = deepcopy(config)

    morphs = nm.load_morphologies([SWC_PATH / 'Neuron.swc', SWC_PATH / 'simple.swc'])
    actual = ms.extract_dataframe(morphs, config)

    assert config == initial_config

    expected = pd.read_csv(Path(DATA_PATH, 'extracted-stats.csv'), header=[0, 1], index_col=0)[
        [
            ("property", "name"),
            ("axon", "max_section_lengths"),
            ("axon", "sum_section_lengths"),
            ("morphology", "mean_soma_radius"),
        ]
    ]
    assert_frame_equal(actual, expected, check_dtype=False)


def test_extract_dataframe_multiproc():
    morphs = [Path(SWC_PATH, name)
            for name in ['Neuron.swc', 'simple.swc']]
    with warnings.catch_warnings(record=True) as w:
        actual = ms.extract_dataframe(morphs, REF_CONFIG, n_workers=2)
        # drop raw features as they require too much test data to mock
        actual = actual.drop(columns='raw_section_branch_orders', level=1)
    expected = pd.read_csv(Path(DATA_PATH, 'extracted-stats.csv'), index_col=0, header=[0, 1])

    assert_frame_equal(actual, expected, check_dtype=False)

    with warnings.catch_warnings(record=True) as w:
        actual = ms.extract_dataframe(morphs, REF_CONFIG, n_workers=os.cpu_count() + 1)
        # drop raw features as they require too much test data to mock
        actual = actual.drop(columns='raw_section_branch_orders', level=1)
        assert len(w) == 1, "Warning not emitted"
    assert_frame_equal(actual, expected, check_dtype=False)


def test_get_header():
    fake_results = {'fake_name0': REF_OUT,
                    'fake_name1': REF_OUT,
                    'fake_name2': REF_OUT,
                    }
    header = ms.get_header(fake_results)
    assert 1 + 2 + 4 * (4 + 5) == len(header)  # name + everything in REF_OUT
    assert 'name' in header
    assert 'morphology:mean_soma_radius' in header


def test_generate_flattened_dict():
    fake_results = {'fake_name0': REF_OUT,
                    'fake_name1': REF_OUT,
                    'fake_name2': REF_OUT,
                    }
    header = ms.get_header(fake_results)
    rows = list(ms.generate_flattened_dict(header, fake_results))
    assert 3 == len(rows)  # one for fake_name[0-2]
    assert 1 + 2 + 4 * (4 + 5) == len(rows[0])  # name + everything in REF_OUT


def test_full_config():
    config = ms.full_config()
    assert set(config.keys()) == {'neurite', 'population', 'morphology', 'neurite_type'}

    assert set(config['neurite'].keys()) == set(_NEURITE_FEATURES.keys())
    assert set(config['morphology'].keys()) == set(_MORPHOLOGY_FEATURES.keys())
    assert set(config['population'].keys()) == set(_POPULATION_FEATURES.keys())


def test_sanitize_config():

    with pytest.raises(ConfigError):
        ms.sanitize_config({'neurite': []})

    new_config = ms.sanitize_config({})  # empty
    assert 2 == len(new_config)  # neurite & morphology created

    full_config = {
        'neurite': {
            'section_lengths': ['max', 'sum'],
            'section_volumes': ['sum'],
            'section_branch_orders': ['max']
        },
        'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL'],
        'morphology': {
            'soma_radius': ['mean']
        }
    }
    new_config = ms.sanitize_config(full_config)
    assert 3 == len(new_config)  # neurite, neurite_type & morphology


def test_multidimensional_features():
    """Features should be split into sub-features when they
    are multidimensional.


    This should be the case even when the feature is `None` or `[]`
    The following morphology has no axon but the axon feature segment_midpoints for
    the axon should still be made of 3 values (X, Y and Z)

    Cf: https://github.com/BlueBrain/NeuroM/issues/859
    """
    m = nm.load_morphology(Path(SWC_PATH, 'no-axon.swc'))

    config = {'neurite': {'segment_midpoints': ['max']},
              'neurite_type': ['AXON']}
    actual = ms.extract_dataframe(m, config)
    assert_array_equal(actual['axon'][['max_segment_midpoints_0',
                                       'max_segment_midpoints_1',
                                       'max_segment_midpoints_2']].values,
                       [[None, None, None]])

    config = {'neurite': {'partition_pairs': ['max']}}
    actual = ms.extract_dataframe(m, config)
    assert_array_equal(actual['axon'][['max_partition_pairs_0',
                                       'max_partition_pairs_1']].values,
                       [[None, None]])
