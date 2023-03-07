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


def test_extract_stats_single_neurite():
    m = nm.load_morphology(SWC_PATH / 'Neuron.swc')
    neurite = m.neurites[0]
    config = deepcopy(REF_CONFIG_NEW)
    config.pop("neurite_type")
    config.pop("morphology")
    res = ms.extract_stats(neurite, config)

    REF_OUT_NEURITE = deepcopy(REF_OUT)
    REF_OUT_NEURITE.pop("morphology", None)
    assert set(res.keys()) == set(REF_OUT_NEURITE.keys())
    assert set(res["axon"].keys()) == set(REF_OUT_NEURITE["axon"].keys())
    for kk in res["axon"].keys():
        assert_almost_equal(res["axon"][kk], REF_OUT_NEURITE["axon"][kk], decimal=4)


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



def test_extract_stats__kwarg_modes_multiple_features():

    m = nm.load_morphology(SWC_PATH / 'Neuron.swc')
    config = {
        'neurite': {
            'principal_direction_extents': {
                'kwargs': [
                    {"direction": 2},
                    {"direction": 1},
                    {"direction": 0},
                ],
                'modes': ['sum', "min"]
            },
        },
        'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL'],
        'morphology': {
            'soma_radius': {'modes': ['mean']},
            'partition_asymmetry': {
                'kwargs': [
                    {'variant': 'branch-order', 'method': 'petilla'},
                    {'variant': 'length', 'method': 'uylings'},
                ],
                'modes': ['min', 'max'],
            },
        }
    }

    res = ms.extract_stats(m, config)

    assert set(res.keys()) == {"axon", "basal_dendrite", "apical_dendrite", "all", "morphology"}

    for key in ("axon", "basal_dendrite", "apical_dendrite", "all"):

        assert set(res[key].keys()) == {
            "sum_principal_direction_extents__direction:2",
            "min_principal_direction_extents__direction:2",
            "sum_principal_direction_extents__direction:1",
            "min_principal_direction_extents__direction:1",
            "sum_principal_direction_extents__direction:0",
            "min_principal_direction_extents__direction:0",
        }

    assert set(res["morphology"].keys()) == {
        "mean_soma_radius",
        "min_partition_asymmetry__variant:branch-order__method:petilla",
        "max_partition_asymmetry__variant:branch-order__method:petilla",
        "min_partition_asymmetry__variant:length__method:uylings",
        "max_partition_asymmetry__variant:length__method:uylings",
    }


def test_extract_dataframe__kwarg_modes_multiple_features():
    m = nm.load_morphology(SWC_PATH / 'Neuron.swc')
    config = {
        'neurite': {
            'principal_direction_extents': {
                'kwargs': [
                    {"direction": 2},
                    {"direction": 1},
                    {"direction": 0},
                ],
                'modes': ['sum', "min"],
            },
        },
        'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL'],
        'morphology': {
            'soma_radius': {'modes': ['mean']},
            'partition_asymmetry': {
                'kwargs': [
                    {'variant': 'branch-order', 'method': 'petilla'},
                    {'variant': 'length', 'method': 'uylings'},
                ],
                'modes': ['min', 'max'],
            },
        },
    }

    res = ms.extract_dataframe(m, config)

    expected_columns = pd.MultiIndex.from_tuples([
        ('property', 'name'),
        ('axon', 'sum_principal_direction_extents__direction:2'),
        ('axon', 'min_principal_direction_extents__direction:2'),
        ('axon', 'sum_principal_direction_extents__direction:1'),
        ('axon', 'min_principal_direction_extents__direction:1'),
        ('axon', 'sum_principal_direction_extents__direction:0'),
        ('axon', 'min_principal_direction_extents__direction:0'),
        ('apical_dendrite', 'sum_principal_direction_extents__direction:2'),
        ('apical_dendrite', 'min_principal_direction_extents__direction:2'),
        ('apical_dendrite', 'sum_principal_direction_extents__direction:1'),
        ('apical_dendrite', 'min_principal_direction_extents__direction:1'),
        ('apical_dendrite', 'sum_principal_direction_extents__direction:0'),
        ('apical_dendrite', 'min_principal_direction_extents__direction:0'),
        ('basal_dendrite', 'sum_principal_direction_extents__direction:2'),
        ('basal_dendrite', 'min_principal_direction_extents__direction:2'),
        ('basal_dendrite', 'sum_principal_direction_extents__direction:1'),
        ('basal_dendrite', 'min_principal_direction_extents__direction:1'),
        ('basal_dendrite', 'sum_principal_direction_extents__direction:0'),
        ('basal_dendrite', 'min_principal_direction_extents__direction:0'),
        ('all', 'sum_principal_direction_extents__direction:2'),
        ('all', 'min_principal_direction_extents__direction:2'),
        ('all', 'sum_principal_direction_extents__direction:1'),
        ('all', 'min_principal_direction_extents__direction:1'),
        ('all', 'sum_principal_direction_extents__direction:0'),
        ('all', 'min_principal_direction_extents__direction:0'),
        ('morphology', 'mean_soma_radius'),
        ('morphology', 'min_partition_asymmetry__variant:branch-order__method:petilla'),
        ('morphology', 'max_partition_asymmetry__variant:branch-order__method:petilla'),
        ('morphology', 'min_partition_asymmetry__variant:length__method:uylings'),
        ('morphology', 'max_partition_asymmetry__variant:length__method:uylings'),
    ])

    pd.testing.assert_index_equal(res.columns, expected_columns)


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
    header = ms._get_header(fake_results)

    assert 1 + 2 + 4 * (4 + 5) == len(header)  # name + everything in REF_OUT
    assert 'name' in header
    assert 'morphology:mean_soma_radius' in header


def test_get_header__with_kwargs():

    fake_results = {
        "fake_name0": {
            'axon': {
                'sum_principal_direction_extents__direction:2': 4.236138323156951,
                'min_principal_direction_extents__direction:2': 4.236138323156951,
                'sum_principal_direction_extents__direction:1': 8.070668782620396,
                'max_principal_direction_extents__direction:1': 8.070668782620396,
                'mean_principal_direction_extents__direction:0': 82.38543140446015
            },
            'apical_dendrite': {
                'sum_principal_direction_extents__direction:2': 3.6493184467335213,
                'min_principal_direction_extents__direction:2': 3.6493184467335213,
                'sum_principal_direction_extents__direction:1': 5.5082642304864695,
                'max_principal_direction_extents__direction:1': 5.5082642304864695,
                'mean_principal_direction_extents__direction:0': 99.57940514500457
            },
            'basal_dendrite': {
                'sum_principal_direction_extents__direction:2': 7.32638745131256,
                'min_principal_direction_extents__direction:2': 3.10141343122575,
                'sum_principal_direction_extents__direction:1': 11.685447149154676,
                'max_principal_direction_extents__direction:1': 6.410958014733595,
                'mean_principal_direction_extents__direction:0': 87.2112016874677
            },
            'all': {
                'sum_principal_direction_extents__direction:2': 15.211844221203034,
                'min_principal_direction_extents__direction:2': 3.10141343122575,
                'sum_principal_direction_extents__direction:1': 25.26438016226154,
                'max_principal_direction_extents__direction:1': 8.070668782620396,
                'mean_principal_direction_extents__direction:0': 89.09680998110002
            },
            'morphology': {
                'mean_soma_radius': 0.13065629977308288,
                'min_partition_asymmetry__variant:branch-order__method:petilla': 0.0,
                'max_partition_asymmetry__variant:branch-order__method:petilla': 0.9,
                'min_partition_asymmetry__variant:length__method:uylings': 0.00030289197373727377,
                'max_partition_asymmetry__variant:length__method:uylings': 0.8795344229855895}
            }
    }

    assert ms._get_header(fake_results) == [
        'name',
        'axon:sum_principal_direction_extents__direction:2',
        'axon:min_principal_direction_extents__direction:2',
        'axon:sum_principal_direction_extents__direction:1',
        'axon:max_principal_direction_extents__direction:1',
        'axon:mean_principal_direction_extents__direction:0',
        'apical_dendrite:sum_principal_direction_extents__direction:2',
        'apical_dendrite:min_principal_direction_extents__direction:2',
        'apical_dendrite:sum_principal_direction_extents__direction:1',
        'apical_dendrite:max_principal_direction_extents__direction:1',
        'apical_dendrite:mean_principal_direction_extents__direction:0',
        'basal_dendrite:sum_principal_direction_extents__direction:2',
        'basal_dendrite:min_principal_direction_extents__direction:2',
        'basal_dendrite:sum_principal_direction_extents__direction:1',
        'basal_dendrite:max_principal_direction_extents__direction:1',
        'basal_dendrite:mean_principal_direction_extents__direction:0',
        'all:sum_principal_direction_extents__direction:2',
        'all:min_principal_direction_extents__direction:2',
        'all:sum_principal_direction_extents__direction:1',
        'all:max_principal_direction_extents__direction:1',
        'all:mean_principal_direction_extents__direction:0',
        'morphology:mean_soma_radius',
        'morphology:min_partition_asymmetry__variant:branch-order__method:petilla',
        'morphology:max_partition_asymmetry__variant:branch-order__method:petilla',
        'morphology:min_partition_asymmetry__variant:length__method:uylings',
        'morphology:max_partition_asymmetry__variant:length__method:uylings'
    ]


def test_generate_flattened_dict():
    fake_results = {'fake_name0': REF_OUT,
                    'fake_name1': REF_OUT,
                    'fake_name2': REF_OUT,
                    }
    header = ms._get_header(fake_results)
    rows = list(ms._generate_flattened_dict(header, fake_results))
    assert 3 == len(rows)  # one for fake_name[0-2]
    assert 1 + 2 + 4 * (4 + 5) == len(rows[0])  # name + everything in REF_OUT


def test_generate_flattened_dict__with_kwargs():

    results = {
        'axon': {
            'sum_principal_direction_extents__direction:2': 0.0,
            'min_principal_direction_extents__direction:2': 1.0,
            'sum_principal_direction_extents__direction:1': 2.0,
            'max_principal_direction_extents__direction:1': 3.0,
            'mean_principal_direction_extents__direction:0': 4.0,
        },
        'apical_dendrite': {
            'sum_principal_direction_extents__direction:2': 5.0,
            'min_principal_direction_extents__direction:2': 6.0,
            'sum_principal_direction_extents__direction:1': 7.0,
            'max_principal_direction_extents__direction:1': 8.0,
            'mean_principal_direction_extents__direction:0': 9.0,
        },
        'basal_dendrite': {
            'sum_principal_direction_extents__direction:2': 1.0,
            'min_principal_direction_extents__direction:2': 2.0,
            'sum_principal_direction_extents__direction:1': 3.0,
            'max_principal_direction_extents__direction:1': 4.0,
            'mean_principal_direction_extents__direction:0': 5.0,
        },
        'all': {
            'sum_principal_direction_extents__direction:2': 6.0,
            'min_principal_direction_extents__direction:2': 7.0,
            'sum_principal_direction_extents__direction:1': 8.0,
            'max_principal_direction_extents__direction:1': 9.0,
            'mean_principal_direction_extents__direction:0': 1.0,
        },
        'morphology': {
            'mean_soma_radius': 2.0,
            'min_partition_asymmetry__variant:branch-order__method:petilla': 3.0,
            'max_partition_asymmetry__variant:branch-order__method:petilla': 4.0,
            'min_partition_asymmetry__variant:length__method:uylings': 5.0,
            'max_partition_asymmetry__variant:length__method:uylings': 6.0,
        }
    }

    fake_results = {
        "fake_name0": results,
        "fake_name1": results,
    }

    header = ms._get_header(fake_results)

    assert list(ms._generate_flattened_dict(header, fake_results)) == [
        [
            'fake_name0', 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [
            'fake_name1', 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ]


def test_full_config():
    config = ms.full_config()
    assert set(config.keys()) == {'neurite', 'population', 'morphology', 'neurite_type'}

    assert set(config['neurite'].keys()) == set(_NEURITE_FEATURES.keys())
    assert set(config['morphology'].keys()) == set(_MORPHOLOGY_FEATURES.keys())
    assert set(config['population'].keys()) == set(_POPULATION_FEATURES.keys())


def test_standardize_layout():
    """Converts the config category entries (e.g. neurite, morphology, population) to using
    the kwarg and modes layout.
    """
    # from short format
    entry = {"f1": ["min", "max"], "f2": ["min"], "f3": []}
    assert ms._standardize_layout(entry) == {
        "f1": {"kwargs": [{}], "modes": ["min", "max"]},
        "f2": {"kwargs": [{}], "modes": ["min"]},
        "f3": {"kwargs": [{}], "modes": []},
    }

    # from kwarg/modes with missing options
    entry = {
        "f1": {"kwargs": {"a1": 1, "a2": 2}, "modes": ["min", "max"]},
        "f2": {"modes": ["min", "median"]},
        "f3": {"kwargs": {"a1": 1, "a2": 2}},
        "f4": {},
    }
    assert ms._standardize_layout(entry) == {
        "f1": {"kwargs": [{"a1": 1, "a2": 2}], "modes": ["min", "max"]},
        "f2": {"kwargs": [{}], "modes": ["min", "median"]},
        "f3": {"kwargs": [{"a1": 1, "a2": 2}], "modes": []},
        "f4": {"kwargs": [{}], "modes": []},
    }

    # from list of kwargs format
    entry = {
        "f1": {"kwargs": [{"a1": 1, "a2": 2}], "modes": ["min", "max"]},
        "f2": {"modes": ["min", "median"]},
        "f3": {"kwargs": [{"a1": 1, "a2": 2}]},
        "f4": {},
    }
    assert ms._standardize_layout(entry) == {
        "f1": {"kwargs": [{"a1": 1, "a2": 2}], "modes": ["min", "max"]},
        "f2": {"kwargs": [{}], "modes": ["min", "median"]},
        "f3": {"kwargs": [{"a1": 1, "a2": 2}], "modes": []},
        "f4": {"kwargs": [{}], "modes": []},
    }


def test_sanitize_config():

    new_config = ms._sanitize_config({})  # empty
    assert 3 == len(new_config)  # neurite & morphology & population created

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
    new_config = ms._sanitize_config(full_config)

    expected_config = {
        'neurite': {
            'section_lengths': {"kwargs": [{}], "modes": ['max', 'sum']},
            'section_volumes': {"kwargs": [{}], "modes": ['sum']},
            'section_branch_orders': {"kwargs": [{}], "modes": ['max']},
        },
        'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL'],
        'morphology': {
            'soma_radius': {"kwargs": [{}], "modes": ["mean"]},
        },
        "population": {},
    }
    assert new_config == expected_config

    # check that legacy neuron entries are converted to morphology ones
    full_config["neuron"] = full_config.pop("morphology")
    assert ms._sanitize_config(full_config) == expected_config

    # check that all formats are converted to the same sanitized config:
    assert ms._sanitize_config(REF_CONFIG) == ms._sanitize_config(REF_CONFIG_NEW)


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
