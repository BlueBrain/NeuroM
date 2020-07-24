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
from pathlib import Path
import warnings

import numpy as np
from nose.tools import (assert_almost_equal, assert_equal,
                        assert_greater_equal, assert_raises, ok_)
import pandas as pd
from pandas.testing import assert_frame_equal

import neurom as nm
from neurom.apps import morph_stats as ms
from neurom.exceptions import ConfigError
from neurom.features import NEURITEFEATURES, NEURONFEATURES


DATA_PATH = Path(__file__).parent.parent.parent.parent / 'test_data'
SWC_PATH = DATA_PATH / 'swc'


REF_CONFIG = {
    'neurite': {
        'section_lengths': ['max', 'total'],
        'section_volumes': ['total'],
        'section_branch_orders': ['max'],
        'segment_midpoints': ['max'],
    },
    'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL'],
    'neuron': {
        'soma_radii': ['mean'],
    }
}

REF_OUT = {
    'mean_soma_radius': 0.13065629648763766,
    'axon': {
        'total_section_length': 207.87975220908129,
        'max_section_length': 11.018460736176685,
        'max_section_branch_order': 10,
        'total_section_volume': 276.73857657289523,
        'max_segment_midpoint_X': 0.0,
        'max_segment_midpoint_Y': 0.0,
        'max_segment_midpoint_Z': 49.520305964149998,
    },
    'all': {
        'total_section_length': 840.68521442251949,
        'max_section_length': 11.758281556059444,
        'max_section_branch_order': 10,
        'total_section_volume': 1104.9077419665782,
        'max_segment_midpoint_X': 64.401674984050004,
        'max_segment_midpoint_Y': 48.48197694465,
        'max_segment_midpoint_Z': 53.750947521650005,
    },
    'apical_dendrite': {
        'total_section_length': 214.37304577550353,
        'max_section_length': 11.758281556059444,
        'max_section_branch_order': 10,
        'total_section_volume': 271.9412385728449,
        'max_segment_midpoint_X': 64.401674984050004,
        'max_segment_midpoint_Y': 0.0,
        'max_segment_midpoint_Z': 53.750947521650005,
    },
    'basal_dendrite': {
        'total_section_length': 418.43241643793476,
        'max_section_length': 11.652508126101711,
        'max_section_branch_order': 10,
        'total_section_volume': 556.22792682083821,
        'max_segment_midpoint_X': 64.007872333250006,
        'max_segment_midpoint_Y': 48.48197694465,
        'max_segment_midpoint_Z': 51.575580778049996,
    },
}


def test_name_correction():
    assert_equal(ms._stat_name('foo', 'raw'), 'foo')
    assert_equal(ms._stat_name('foos', 'raw'), 'foo')
    assert_equal(ms._stat_name('foos', 'bar'), 'bar_foo')
    assert_equal(ms._stat_name('foos', 'total'), 'total_foo')
    assert_equal(ms._stat_name('soma_radii', 'total'), 'total_soma_radius')
    assert_equal(ms._stat_name('soma_radii', 'raw'), 'soma_radius')


def test_eval_stats_raw_returns_list():
    assert_equal(ms.eval_stats(np.array([1, 2, 3, 4]), 'raw'), [1, 2, 3, 4])


def test_eval_stats_empty_input_returns_none():
    ok_(ms.eval_stats([], 'min') is None)


def test_eval_stats_total_returns_sum():
    assert_equal(ms.eval_stats(np.array([1, 2, 3, 4]), 'total'), 10)


def test_eval_stats_applies_numpy_function():
    modes = ('min', 'max', 'mean', 'median', 'std')

    ref_array = np.arange(1, 10)

    for m in modes:
        assert_equal(ms.eval_stats(ref_array, m),
            getattr(np, m)(ref_array))


def test_extract_stats_single_neuron():
    nrn = nm.load_neuron(Path(SWC_PATH, 'Neuron.swc'))
    res = ms.extract_stats(nrn, REF_CONFIG)
    assert_equal(set(res.keys()), set(REF_OUT.keys()))
    # Note: soma radius is calculated from the sphere that gives the area
    # of the cylinders described in Neuron.swc
    assert_almost_equal(res['mean_soma_radius'], REF_OUT['mean_soma_radius'])

    for k in ('all', 'axon', 'basal_dendrite', 'apical_dendrite'):
        assert_equal(set(res[k].keys()), set(REF_OUT[k].keys()))
        for kk in res[k].keys():
            assert_almost_equal(res[k][kk], REF_OUT[k][kk], places=3)


def test_extract_dataframe():
    # Vanilla test
    nrns = nm.load_neurons([Path(SWC_PATH, name)
                            for name in ['Neuron.swc', 'simple.swc']])
    actual = ms.extract_dataframe(nrns, REF_CONFIG)
    expected = pd.read_csv(Path(DATA_PATH, 'extracted-stats.csv'), index_col=0)
    assert_frame_equal(actual, expected)

    # Test with a single neuron in the population
    nrns = nm.load_neurons(Path(SWC_PATH, 'Neuron.swc'))
    actual = ms.extract_dataframe(nrns, REF_CONFIG)
    assert_frame_equal(actual, expected[expected.name == 'Neuron'], check_dtype=False)

    # Test with a config without the 'neuron' key
    nrns = nm.load_neurons([Path(SWC_PATH, name)
                            for name in ['Neuron.swc', 'simple.swc']])
    config = {'neurite': {'section_lengths': ['total']},
              'neurite_type': ['AXON', 'APICAL_DENDRITE', 'BASAL_DENDRITE', 'ALL']}
    actual = ms.extract_dataframe(nrns, config)
    expected = expected[['name', 'neurite_type', 'total_section_length']]
    assert_frame_equal(actual, expected)

    # Test with a FstNeuron argument
    nrn = nm.load_neuron(Path(SWC_PATH, 'Neuron.swc'))
    actual = ms.extract_dataframe(nrn, config)
    assert_frame_equal(actual, expected[expected.name == 'Neuron'], check_dtype=False)

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
    expected = pd.DataFrame(
        columns=['name', 'neurite_type', 'total_total_length_per_neurite'],
        data=[['Neuron', 'axon', 207.879752],
              ['Neuron', 'basal_dendrite', 418.432416],
              ['Neuron', 'apical_dendrite', 214.373046],
              ['Neuron', 'all', 840.685214],
              ['simple', 'axon', 15.000000],
              ['simple', 'basal_dendrite', 16.000000],
              ['simple', 'apical_dendrite', 0.000000],
              ['simple', 'all', 31.000000]])
    assert_frame_equal(actual, expected)


def test_extract_dataframe_multiproc():
    nrns = nm.load_neurons([Path(SWC_PATH, name)
                            for name in ['Neuron.swc', 'simple.swc']])
    with warnings.catch_warnings(record=True) as w:
        actual = ms.extract_dataframe(nrns, REF_CONFIG, n_workers=2)
    expected = pd.read_csv(Path(DATA_PATH, 'extracted-stats.csv'), index_col=0)

    # Compare sorted DataFrame since Pool.imap_unordered disrupted the order
    assert_frame_equal(actual.sort_values(by=['name']).reset_index(drop=True),
                       expected.sort_values(by=['name']).reset_index(drop=True))

    with warnings.catch_warnings(record=True) as w:
        actual = ms.extract_dataframe(nrns, REF_CONFIG, n_workers=os.cpu_count() + 1)
        assert_equal(len(w), 1, "Warning not emitted")
    assert_frame_equal(actual.sort_values(by=['name']).reset_index(drop=True),
                       expected.sort_values(by=['name']).reset_index(drop=True))



def test_get_header():
    fake_results = {'fake_name0': REF_OUT,
                    'fake_name1': REF_OUT,
                    'fake_name2': REF_OUT,
                    }
    header = ms.get_header(fake_results)
    assert_equal(1 + 1 + 4 * (4 + 3), len(header))  # name + everything in REF_OUT
    ok_('name' in header)
    ok_('mean_soma_radius' in header)


def test_generate_flattened_dict():
    fake_results = {'fake_name0': REF_OUT,
                    'fake_name1': REF_OUT,
                    'fake_name2': REF_OUT,
                    }
    header = ms.get_header(fake_results)
    rows = list(ms.generate_flattened_dict(header, fake_results))
    assert_equal(3, len(rows))  # one for fake_name[0-2]
    assert_equal(1 + 1 + 4 * (4 + 3), len(rows[0]))  # name + everything in REF_OUT


def test_full_config():
    config = ms.full_config()
    assert_equal(set(config.keys()), {'neurite', 'neuron', 'neurite_type'})

    assert_equal(set(config['neurite'].keys()), set(NEURITEFEATURES.keys()))
    assert_equal(set(config['neuron'].keys()), set(NEURONFEATURES.keys()))


def test_sanitize_config():
    assert_raises(ConfigError, ms.sanitize_config, {'neurite': []})

    new_config = ms.sanitize_config({})  # empty
    assert_equal(2, len(new_config))  # neurite & neuron created

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
    assert_equal(3, len(new_config))  # neurite, neurite_type & neuron
