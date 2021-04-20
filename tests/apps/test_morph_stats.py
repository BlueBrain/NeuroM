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
from pathlib import Path

import neurom as nm
import numpy as np
import pandas as pd
from neurom.apps import morph_stats as ms
from neurom.exceptions import ConfigError
from neurom.features import NEURITEFEATURES, NEURONFEATURES

import pytest
from numpy.testing import assert_array_equal, assert_almost_equal
from pandas.testing import assert_frame_equal

DATA_PATH = Path(__file__).parent.parent / 'data'
SWC_PATH = DATA_PATH / 'swc'
REF_CONFIG = {
    'neurite': {
        'section_lengths': ['max', 'total'],
        'section_volumes': ['total'],
        'section_branch_orders': ['max'],
        'segment_midpoints': ['max'],
        'max_radial_distance': ['mean'],
    },
    'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL'],
    'neuron': {
        'soma_radii': ['mean'],
        'max_radial_distance': ['mean'],
    }
}

REF_OUT = {
    'neuron': {
        'mean_soma_radius': 0.13065629648763766,
        'mean_max_radial_distance': 99.5894610648815,
    },
    'axon': {
        'total_section_length': 207.87975220908129,
        'max_section_length': 11.018460736176685,
        'max_section_branch_order': 10,
        'total_section_volume': 276.73857657289523,
        'max_segment_midpoint_0': 0.0,
        'max_segment_midpoint_1': 0.0,
        'max_segment_midpoint_2': 49.520305964149998,
        'mean_max_radial_distance': 82.44254511788921,
    },
    'all': {
        'total_section_length': 840.68521442251949,
        'max_section_length': 11.758281556059444,
        'max_section_branch_order': 10,
        'total_section_volume': 1104.9077419665782,
        'max_segment_midpoint_0': 64.401674984050004,
        'max_segment_midpoint_1': 48.48197694465,
        'max_segment_midpoint_2': 53.750947521650005,
        'mean_max_radial_distance': 99.5894610648815,
    },
    'apical_dendrite': {
        'total_section_length': 214.37304577550353,
        'max_section_length': 11.758281556059444,
        'max_section_branch_order': 10,
        'total_section_volume': 271.9412385728449,
        'max_segment_midpoint_0': 64.401674984050004,
        'max_segment_midpoint_1': 0.0,
        'max_segment_midpoint_2': 53.750947521650005,
        'mean_max_radial_distance': 99.5894610648815,
    },
    'basal_dendrite': {
        'total_section_length': 418.43241643793476,
        'max_section_length': 11.652508126101711,
        'max_section_branch_order': 10,
        'total_section_volume': 556.22792682083821,
        'max_segment_midpoint_0': 64.007872333250006,
        'max_segment_midpoint_1': 48.48197694465,
        'max_segment_midpoint_2': 51.575580778049996,
        'mean_max_radial_distance': 94.43342438865741,
    },
}


def test_name_correction():
    assert ms._stat_name('foo', 'raw') == 'foo'
    assert ms._stat_name('foos', 'raw') == 'foo'
    assert ms._stat_name('foos', 'bar') == 'bar_foo'
    assert ms._stat_name('foos', 'total') == 'total_foo'
    assert ms._stat_name('soma_radii', 'total') == 'total_soma_radius'
    assert ms._stat_name('soma_radii', 'raw') == 'soma_radius'


def test_eval_stats_raw_returns_list():
    assert ms.eval_stats(np.array([1, 2, 3, 4]), 'raw') == [1, 2, 3, 4]


def test_eval_stats_empty_input_returns_none():
    assert ms.eval_stats([], 'min') is None


def test_eval_stats_total_returns_sum():
    assert ms.eval_stats(np.array([1, 2, 3, 4]), 'total') == 10


def test_eval_stats_on_empty_stat():
    assert ms.eval_stats(np.array([]), 'mean') == None
    assert ms.eval_stats(np.array([]), 'std') == None
    assert ms.eval_stats(np.array([]), 'median') == None
    assert ms.eval_stats(np.array([]), 'min') == None
    assert ms.eval_stats(np.array([]), 'max') == None

    assert ms.eval_stats(np.array([]), 'raw') == []
    assert ms.eval_stats(np.array([]), 'total') == 0.0


def test_eval_stats_applies_numpy_function():
    modes = ('min', 'max', 'mean', 'median', 'std')

    ref_array = np.arange(1, 10)

    for m in modes:
        assert (ms.eval_stats(ref_array, m) == getattr(np, m)(ref_array))


def test_extract_stats_single_neuron():
    nrn = nm.load_neuron(SWC_PATH / 'Neuron.swc')
    res = ms.extract_stats(nrn, REF_CONFIG)
    assert set(res.keys()) == set(REF_OUT.keys())
    for k in ('neuron', 'all', 'axon', 'basal_dendrite', 'apical_dendrite'):
        assert set(res[k].keys()) == set(REF_OUT[k].keys())
        for kk in res[k].keys():
            assert_almost_equal(res[k][kk], REF_OUT[k][kk], decimal=4)


def test_extract_stats_scalar_feature():
    nrn = nm.load_neuron(DATA_PATH / 'neurolucida' / 'bio_neuron-000.asc')
    config = {
        'neurite_type': ['ALL'],
        'neurite': {
            'n_forking_points': ['max'],
        },
        'neuron': {
            'soma_volume': ['total'],
        }
    }
    res = ms.extract_stats(nrn, config)
    assert res == {'all': {'max_n_forking_point': 277},
                       'neuron': {'total_soma_volume': 1424.4383771584492}}


def test_extract_dataframe():
    # Vanilla test
    nrns = nm.load_neurons([SWC_PATH / name
                            for name in ['Neuron.swc', 'simple.swc']])
    actual = ms.extract_dataframe(nrns, REF_CONFIG)
    expected = pd.read_csv(Path(DATA_PATH, 'extracted-stats.csv'), header=[0, 1], index_col=0)
    assert_frame_equal(actual, expected)

    # Test with a single neuron in the population
    nrns = nm.load_neurons(Path(SWC_PATH, 'Neuron.swc'))
    actual = ms.extract_dataframe(nrns, REF_CONFIG)
    assert_frame_equal(actual, expected.iloc[[0]], check_dtype=False)

    # Test with a config without the 'neuron' key
    nrns = nm.load_neurons([Path(SWC_PATH, name)
                            for name in ['Neuron.swc', 'simple.swc']])
    config = {'neurite': {'section_lengths': ['total']},
              'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL']}
    actual = ms.extract_dataframe(nrns, config)
    idx = pd.IndexSlice
    expected = expected.loc[:, idx[:, ['name', 'total_section_length']]]
    assert_frame_equal(actual, expected)

    # Test with a FstNeuron argument
    nrn = nm.load_neuron(Path(SWC_PATH, 'Neuron.swc'))
    actual = ms.extract_dataframe(nrn, config)
    assert_frame_equal(actual, expected.iloc[[0]], check_dtype=False)

    # Test with a List[FstNeuron] argument
    nrns = [nm.load_neuron(Path(SWC_PATH, name))
            for name in ['Neuron.swc', 'simple.swc']]
    actual = ms.extract_dataframe(nrns, config)
    assert_frame_equal(actual, expected)

    # Test with a List[Path] argument
    nrns = [Path(SWC_PATH, name) for name in ['Neuron.swc', 'simple.swc']]
    actual = ms.extract_dataframe(nrns, config)
    assert_frame_equal(actual, expected)

    # Test without any neurite_type keys, it should pick the defaults
    config = {'neurite': {'total_length_per_neurite': ['total']}}
    actual = ms.extract_dataframe(nrns, config)
    expected_columns = pd.MultiIndex.from_tuples(
        [('property', 'name'),
         ('axon', 'total_total_length_per_neurite'),
         ('basal_dendrite', 'total_total_length_per_neurite'),
         ('apical_dendrite', 'total_total_length_per_neurite'),
         ('all', 'total_total_length_per_neurite')])
    expected = pd.DataFrame(
        columns=expected_columns,
        data=[['Neuron', 207.87975221, 418.43241644, 214.37304578, 840.68521442],
              ['simple', 15.,          16.,           0.,          31., ]])
    assert_frame_equal(actual, expected)


def test_extract_dataframe_multiproc():
    # FIXME: Cannot use Neuron objects in the extract_dataframe ctor right now
    # because of "TypeError: can't pickle Neuron objects"
    # nrns = nm.load_neurons([Path(SWC_PATH, name)
    #                         for name in ['Neuron.swc', 'simple.swc']])
    nrns = [Path(SWC_PATH, name)
            for name in ['Neuron.swc', 'simple.swc']]
    with warnings.catch_warnings(record=True) as w:
        actual = ms.extract_dataframe(nrns, REF_CONFIG, n_workers=2)
    expected = pd.read_csv(Path(DATA_PATH, 'extracted-stats.csv'), index_col=0, header=[0, 1])

    assert_frame_equal(actual, expected)

    with warnings.catch_warnings(record=True) as w:
        actual = ms.extract_dataframe(nrns, REF_CONFIG, n_workers=os.cpu_count() + 1)
        assert len(w) == 1, "Warning not emitted"
    assert_frame_equal(actual, expected)


def test_get_header():
    fake_results = {'fake_name0': REF_OUT,
                    'fake_name1': REF_OUT,
                    'fake_name2': REF_OUT,
                    }
    header = ms.get_header(fake_results)
    assert 1 + 2 + 4 * (4 + 4) == len(header)  # name + everything in REF_OUT
    assert 'name' in header
    assert 'neuron:mean_soma_radius' in header


def test_generate_flattened_dict():
    fake_results = {'fake_name0': REF_OUT,
                    'fake_name1': REF_OUT,
                    'fake_name2': REF_OUT,
                    }
    header = ms.get_header(fake_results)
    rows = list(ms.generate_flattened_dict(header, fake_results))
    assert 3 == len(rows)  # one for fake_name[0-2]
    assert 1 + 2 + 4 * (4 + 4) == len(rows[0])  # name + everything in REF_OUT


def test_full_config():
    config = ms.full_config()
    assert set(config.keys()) == {'neurite', 'neuron', 'neurite_type'}

    assert set(config['neurite'].keys()) == set(NEURITEFEATURES.keys())
    assert set(config['neuron'].keys()) == set(NEURONFEATURES.keys())


def test_sanitize_config():

    with pytest.raises(ConfigError):
        ms.sanitize_config({'neurite': []})

    new_config = ms.sanitize_config({})  # empty
    assert 2 == len(new_config)  # neurite & neuron created

    full_config = {
        'neurite': {
            'section_lengths': ['max', 'total'],
            'section_volumes': ['total'],
            'section_branch_orders': ['max']
        },
        'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL'],
        'neuron': {
            'soma_radii': ['mean']
        }
    }
    new_config = ms.sanitize_config(full_config)
    assert 3 == len(new_config)  # neurite, neurite_type & neuron


def test_multidimensional_features():
    """Features should be split into sub-features when they
    are multidimensional.


    This should be the case even when the feature is `None` or `[]`
    The following neuron has no axon but the axon feature segment_midpoints for
    the axon should still be made of 3 values (X, Y and Z)

    Cf: https://github.com/BlueBrain/NeuroM/issues/859
    """
    neuron = nm.load_neuron(Path(SWC_PATH, 'no-axon.swc'))

    config = {'neurite': {'segment_midpoints': ['max']},
              'neurite_type': ['AXON']}
    actual = ms.extract_dataframe(neuron, config)
    assert_array_equal(actual['axon'][['max_segment_midpoint_0',
                                       'max_segment_midpoint_1',
                                       'max_segment_midpoint_2']].values,
                       [[None, None, None]])

    config = {'neurite': {'partition_pairs': ['max']}}
    actual = ms.extract_dataframe(neuron, config)
    assert_array_equal(actual['axon'][['max_partition_pair_0',
                                       'max_partition_pair_1']].values,
                       [[None, None]])
