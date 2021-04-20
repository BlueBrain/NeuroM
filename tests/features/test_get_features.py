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

"""Test neurom.features_get features."""

import math
from io import StringIO
from pathlib import Path

import neurom as nm
import numpy as np
from mock import patch
from neurom import core, features, iter_neurites, iter_sections, load_neuron, load_neurons
from neurom.core.population import Population
from neurom.core.types import NeuriteType
from neurom.core.types import tree_type_checker as _is_type
from neurom.exceptions import NeuroMError
from neurom.features import NEURITEFEATURES
from neurom.features import get as get_feature
from neurom.features import neuritefunc as nf

import pytest
from numpy.testing import assert_allclose

DATA_PATH = Path(__file__).parent.parent / 'data'
NRN_FILES = [DATA_PATH / 'h5/v1' / f
             for f in ('Neuron.h5', 'Neuron_2_branch.h5', 'bio_neuron-001.h5')]
NRNS = load_neurons(NRN_FILES)
NRN = NRNS[0]
POP = Population(NRNS)

SWC_PATH = DATA_PATH / 'swc'
NEURON_PATH = SWC_PATH / 'Neuron.swc'
NEURON = load_neuron(NEURON_PATH)
NEURITES = (NeuriteType.axon,
            NeuriteType.apical_dendrite,
            NeuriteType.basal_dendrite,
            NeuriteType.all)


def assert_items_equal(a, b):
    assert sorted(a) == sorted(b)


def assert_features_for_neurite(feat, neurons, expected, exact=True):
    for neurite_type, expected_values in expected.items():
        if neurite_type is None:
            res_pop = get_feature(feat, neurons)
            res = get_feature(feat, neurons[0])
        else:
            res_pop = get_feature(feat, neurons, neurite_type=neurite_type)
            res = get_feature(feat, neurons[0], neurite_type=neurite_type)

        if exact:
            assert_items_equal(res_pop, expected_values)
        else:
            assert_allclose(res_pop, expected_values)

        # test for single neuron
        if isinstance(res, np.ndarray):
            # some features, ex: total_length return arrays w/ one element when
            # called on a single neuron
            assert len(res) == 1
            res = res[0]
        if exact:
            assert res == expected_values[0]
        else:
            assert_allclose(res, expected_values[0])


def _stats(seq):
    return np.min(seq), np.max(seq), np.sum(seq), np.mean(seq)


def test_number_of_sections():
    feat = 'number_of_sections'
    expected = {None: [84, 42, 202],
                NeuriteType.all: [84, 42, 202],
                NeuriteType.axon: [21, 21, 179],
                NeuriteType.apical_dendrite: [21, 0, 0],
                NeuriteType.basal_dendrite: [42, 21, 23],
                }
    assert_features_for_neurite(feat, POP, expected)


def test_max_radial_distances():
    feat = 'max_radial_distances'
    expected = {
        None: [99.58945832, 94.43342439, 1053.77939245],
        NeuriteType.all: [99.58945832, 94.43342439, 1053.77939245],
        NeuriteType.axon: [82.442545, 82.442545, 1053.779392],
        NeuriteType.basal_dendrite: [94.43342563, 94.43342439, 207.56977859],
    }
    assert_features_for_neurite(feat, POP, expected, exact=False)

    # Test with a list of neurites
    neurites = POP[0].neurites
    expected = {
        None: [99.58945832],
        NeuriteType.all: [99.58945832],
        NeuriteType.apical_dendrite: [99.589458],
    }
    assert_features_for_neurite(feat, neurites, expected, exact=False)


def test_max_radial_distance():
    feat = 'max_radial_distance'
    neurites = POP[0].neurites
    expected = {
        None: 99.58945832,
        NeuriteType.all: 99.58945832,
        NeuriteType.apical_dendrite: 99.589458,
    }

    for neurite_type, expected_values in expected.items():
        if neurite_type is None:
            res = get_feature(feat, neurites)
        else:
            res = get_feature(feat, neurites, neurite_type=neurite_type)
        assert_allclose(res, expected_values)


def test_section_tortuosity_pop():

    feat = 'section_tortuosity'

    assert_allclose(_stats(get_feature(feat, POP)),
                    (1.0,
                     4.657,
                     440.408,
                     1.342),
                    rtol=1e-3)

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.all)),
                    (1.0,
                     4.657,
                     440.408,
                     1.342),
                    rtol=1e-3)

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                    (1.070,
                     1.573,
                     26.919,
                     1.281),
                    rtol=1e-3)

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                    (1.042,
                     1.674,
                     106.596,
                     1.239),
                    rtol=1e-3)


def test_section_tortuosity_nrn():
    feat = 'section_tortuosity'
    assert_allclose(_stats(get_feature(feat, NRN)),
                    (1.070,
                     1.573,
                     106.424,
                     1.266),
                    rtol=1e-3)

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.all)),
                    (1.070,
                     1.573,
                     106.424,
                     1.266),
                    rtol=1e-3)

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                    (1.070,
                     1.573,
                     26.919,
                     1.281),
                    rtol=1e-3)

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                    (1.078,
                     1.550,
                     51.540,
                     1.227),
                    rtol=1e-3)


def test_number_of_segments():

    feat = 'number_of_segments'

    expected = {None: [840, 419, 5179],
                NeuriteType.all: [840, 419, 5179],
                NeuriteType.axon: [210, 209, 4508],
                NeuriteType.apical_dendrite: [210, 0, 0],
                NeuriteType.basal_dendrite: [420, 210, 671],
                }

    assert_features_for_neurite(feat, POP, expected)


def test_number_of_neurites_pop():
    feat = 'number_of_neurites'
    expected = {None: [4, 2, 4],
                NeuriteType.all: [4, 2, 4],
                NeuriteType.axon: [1, 1, 1],
                NeuriteType.apical_dendrite: [1, 0, 0],
                NeuriteType.basal_dendrite: [2, 1, 3],
                }
    assert_features_for_neurite(feat, POP, expected)


def test_number_of_bifurcations_pop():
    feat = 'number_of_bifurcations'
    expected = {None: [40, 20, 97],
                NeuriteType.all: [40, 20, 97],
                NeuriteType.axon: [10, 10, 87],
                NeuriteType.apical_dendrite: [10, 0, 0],
                NeuriteType.basal_dendrite: [20, 10, 10],
                }
    assert_features_for_neurite(feat, POP, expected)


def test_number_of_forking_points_pop():

    feat = 'number_of_forking_points'

    expected = {None: [40, 20, 98],
                NeuriteType.all: [40, 20, 98],
                NeuriteType.axon: [10, 10, 88],
                NeuriteType.apical_dendrite: [10, 0, 0],
                NeuriteType.basal_dendrite: [20, 10, 10],
                }
    assert_features_for_neurite(feat, POP, expected)


def test_number_of_terminations_pop():
    feat = 'number_of_terminations'
    expected = {None: [44, 22, 103],
                NeuriteType.all: [44, 22, 103],
                NeuriteType.axon: [11, 11, 90],
                NeuriteType.apical_dendrite: [11, 0, 0],
                NeuriteType.basal_dendrite: [22, 11, 13],
                }
    assert_features_for_neurite(feat, POP, expected)


def test_total_length_pop():
    feat = 'total_length'
    expected = {None: [840.68522362011538, 418.83424432013902, 13250.825773939932],
                NeuriteType.all: [840.68522362011538, 418.83424432013902, 13250.825773939932],
                NeuriteType.axon: [207.8797736031714, 207.81088341560977, 11767.156115224638],
                NeuriteType.apical_dendrite: [214.37302709169489, 0, 0],
                NeuriteType.basal_dendrite: [418.43242292524889, 211.02336090452931, 1483.6696587152967],
                }
    assert_features_for_neurite(feat, POP, expected, exact=False)


def test_segment_radii_pop():

    feat = 'segment_radii'

    assert_allclose(_stats(get_feature(feat, POP)),
                    (0.079999998211860657,
                        1.2150000333786011,
                        1301.9191725363567,
                        0.20222416473071708))

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.all)),
                    (0.079999998211860657,
                        1.2150000333786011,
                        1301.9191725363567,
                        0.20222416473071708))

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                    (0.13142434507608414,
                        1.0343990325927734,
                        123.41135908663273,
                        0.58767313850777492))

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                    (0.079999998211860657,
                        1.2150000333786011,
                        547.43900821779164,
                        0.42078324997524336))


def test_segment_radii_nrn():

    feat = 'segment_radii'

    assert_allclose(_stats(get_feature(feat, NRN)),
                    (0.12087134271860123,
                        1.0343990325927734,
                        507.01994501426816,
                        0.60359517263603357))

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.all)),
                    (0.12087134271860123,
                        1.0343990325927734,
                        507.01994501426816,
                        0.60359517263603357))

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                    (0.13142434507608414,
                        1.0343990325927734,
                        123.41135908663273,
                        0.58767313850777492))

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                    (0.14712842553853989,
                        1.0215770602226257,
                        256.71241207793355,
                        0.61122002875698467))


def test_segment_meander_angles_pop():

    feat = 'segment_meander_angles'

    assert_allclose(_stats(get_feature(feat, POP)),
                    (0.0, 3.1415, 14637.9776, 2.3957),
                    rtol=1e-3)

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.all)),
                    (0.0, 3.1415, 14637.9776, 2.3957),
                    rtol=1e-3)

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                    (0.3261, 3.0939, 461.9816, 2.4443),
                    rtol=1e-4)

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                    (0.0, 3.1415, 2926.2411, 2.4084),
                    rtol=1e-4)


def test_segment_meander_angles_nrn():

    feat = 'segment_meander_angles'
    assert_allclose(_stats(get_feature(feat, NRN)),
                    (0.32610, 3.12996, 1842.35, 2.43697),
                    rtol=1e-5)

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.all)),
                    (0.32610, 3.12996, 1842.35, 2.43697),
                    rtol=1e-5)

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                    (0.32610, 3.09392, 461.981, 2.44434),
                    rtol=1e-5)

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                    (0.47318, 3.12996, 926.338, 2.45063),
                    rtol=1e-4)


def test_neurite_volumes_nrn():

    feat = 'neurite_volumes'

    assert_allclose(_stats(get_feature(feat, NRN)),
                    (271.9412, 281.2475, 1104.907, 276.2269),
                    rtol=1e-5)

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.all)),
                    (271.9412, 281.2475, 1104.907, 276.2269),
                    rtol=1e-5)

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.axon)),
                    (276.7386, 276.7386, 276.7386, 276.7386),
                    rtol=1e-5)

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                    (274.9803, 281.2475, 556.2279, 278.1139),
                    rtol=1e-5)

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                    (271.9412, 271.9412, 271.9412, 271.9412),
                    rtol=1e-5)


def test_neurite_volumes_pop():

    feat = 'neurite_volumes'

    assert_allclose(_stats(get_feature(feat, POP)),
                    (28.356406629821159, 281.24754646913954, 2249.4613918388391, 224.9461391838839))

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.all)),
                    (28.356406629821159, 281.24754646913954, 2249.4613918388391, 224.9461391838839))

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.axon)),
                    (276.58135508666612, 277.5357232437392, 830.85568094763551, 276.95189364921185))

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                    (28.356406629821159, 281.24754646913954, 1146.6644894516851, 191.1107482419475))

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                    (271.94122143951864, 271.94122143951864, 271.94122143951864, 271.94122143951864))


def test_neurite_density_nrn():

    feat = 'neurite_volume_density'

    assert_allclose(_stats(get_feature(feat, NRN)),
                    (0.24068543213643726, 0.52464681266899216, 1.4657913638494682, 0.36644784096236704))

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.all)),
                    (0.24068543213643726, 0.52464681266899216, 1.4657913638494682, 0.36644784096236704))

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.axon)),
                    (0.26289304906104355, 0.26289304906104355, 0.26289304906104355, 0.26289304906104355))

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                    (0.24068543213643726, 0.52464681266899216, 0.76533224480542938, 0.38266612240271469))

    assert_allclose(_stats(get_feature(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                    (0.43756606998299519, 0.43756606998299519, 0.43756606998299519, 0.43756606998299519))


def test_neurite_density_pop():

    feat = 'neurite_volume_density'

    assert_allclose(_stats(get_feature(feat, POP)),
                    (6.1847539631150784e-06, 0.52464681266899216, 1.9767794901940539, 0.19767794901940539))

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.all)),
                    (6.1847539631150784e-06, 0.52464681266899216, 1.9767794901940539, 0.19767794901940539))

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.axon)),
                    (6.1847539631150784e-06, 0.26465213325053372, 0.5275513670655404, 0.17585045568851346),
                    rtol=1e-6)

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                    (0.00034968816544949771, 0.52464681266899216, 1.0116620531455183, 0.16861034219091972))

    assert_allclose(_stats(get_feature(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                    (0.43756606998299519, 0.43756606998299519, 0.43756606998299519, 0.43756606998299519))


def test_segment_meander_angles_single_section():
    feat = 'segment_meander_angles'

    nrn = nm.load_neuron(StringIO(u"""((CellBody) (0 0 0 0))
                                      ((Dendrite)
                                       (0 0 0 2)
                                       (1 0 0 2)
                                       (1 1 0 2)
                                       (2 1 0 2)
                                       (2 2 0 2)))"""), reader='asc')

    nrt = nrn.neurites[0]
    pop = core.Population([nrn])

    ref = [math.pi / 2, math.pi / 2, math.pi / 2]

    assert ref == get_feature(feat, nrt).tolist()
    assert ref == get_feature(feat, nrn).tolist()
    assert ref == get_feature(feat, pop).tolist()


def test_neurite_features_accept_single_tree():
    for f in NEURITEFEATURES:
        ret = get_feature(f, NRN.neurites[0])
        if isinstance(ret, np.ndarray):
            assert ret.dtype.kind in ('i', 'f')
            assert len(features._find_feature_func(f).shape) >= 1
            assert len(ret) > 0
        else:
            assert np.isscalar(ret)


@patch.dict(NEURITEFEATURES)
def test_register_neurite_feature_nrns():

    def npts(neurite):
        return len(neurite.points)

    def vol(neurite):
        return neurite.volume

    features.register_neurite_feature('foo', npts)

    n_points_ref = [len(n.points) for n in iter_neurites(NRNS)]
    n_points = get_feature('foo', NRNS)
    assert_items_equal(n_points, n_points_ref)

    # test neurite type filtering
    n_points_ref = [len(n.points) for n in iter_neurites(NRNS, filt=_is_type(NeuriteType.axon))]
    n_points = get_feature('foo', NRNS, neurite_type=NeuriteType.axon)
    assert_items_equal(n_points, n_points_ref)

    features.register_neurite_feature('bar', vol)

    n_volume_ref = [n.volume for n in iter_neurites(NRNS)]
    n_volume = get_feature('bar', NRNS)
    assert_items_equal(n_volume, n_volume_ref)

    # test neurite type filtering
    n_volume_ref = [n.volume for n in iter_neurites(NRNS, filt=_is_type(NeuriteType.axon))]
    n_volume = get_feature('bar', NRNS, neurite_type=NeuriteType.axon)
    assert_items_equal(n_volume, n_volume_ref)


@patch.dict(NEURITEFEATURES)
def test_register_neurite_feature_pop():

    def npts(neurite):
        return len(neurite.points)

    def vol(neurite):
        return neurite.volume

    features.register_neurite_feature('foo', npts)

    n_points_ref = [len(n.points) for n in iter_neurites(POP)]
    n_points = get_feature('foo', POP)
    assert_items_equal(n_points, n_points_ref)

    # test neurite type filtering
    n_points_ref = [len(n.points) for n in iter_neurites(POP,
                                                         filt=_is_type(NeuriteType.basal_dendrite))]
    n_points = get_feature('foo', POP, neurite_type=NeuriteType.basal_dendrite)
    assert_items_equal(n_points, n_points_ref)

    features.register_neurite_feature('bar', vol)

    n_volume_ref = [n.volume for n in iter_neurites(POP)]
    n_volume = get_feature('bar', POP)
    assert_items_equal(n_volume, n_volume_ref)

    # test neurite type filtering
    n_volume_ref = [n.volume for n in iter_neurites(POP, filt=_is_type(NeuriteType.basal_dendrite))]
    n_volume = get_feature('bar', POP, neurite_type=NeuriteType.basal_dendrite)
    assert_items_equal(n_volume, n_volume_ref)


def test_get_raises():
    with pytest.raises(NeuroMError):
        get_feature('ahah-I-do-not-exist!', lambda n: None)


def test_register_existing_feature_raises():
    with pytest.raises(NeuroMError):
        features.register_neurite_feature('total_length', lambda n: None)


def test_section_lengths():
    ref_seclen = [n.length for n in iter_sections(NEURON)]
    seclen = get_feature('section_lengths', NEURON)
    assert len(seclen) == 84
    assert_allclose(seclen, ref_seclen)


def test_section_lengths_axon():
    s = get_feature('section_lengths', NEURON, neurite_type=NeuriteType.axon)
    assert len(s) == 21


def test_total_lengths_basal():
    s = get_feature('section_lengths', NEURON, neurite_type=NeuriteType.basal_dendrite)
    assert len(s) == 42


def test_section_lengths_apical():
    s = get_feature('section_lengths', NEURON, neurite_type=NeuriteType.apical_dendrite)
    assert len(s) == 21


def test_total_length_per_neurite_axon():
    tl = get_feature('total_length_per_neurite', NEURON, neurite_type=NeuriteType.axon)
    assert len(tl) == 1
    assert_allclose(tl, (207.87975221,))


def test_total_length_per_neurite_basal():
    tl = get_feature('total_length_per_neurite', NEURON, neurite_type=NeuriteType.basal_dendrite)
    assert len(tl) == 2
    assert_allclose(tl, (211.11737442, 207.31504202))


def test_total_length_per_neurite_apical():
    tl = get_feature('total_length_per_neurite', NEURON, neurite_type=NeuriteType.apical_dendrite)
    assert len(tl) == 1
    assert_allclose(tl, (214.37304578,))


def test_total_length_axon():
    tl = get_feature('total_length', NEURON, neurite_type=NeuriteType.axon)
    assert len(tl) == 1
    assert_allclose(tl, (207.87975221,))


def test_total_length_basal():
    tl = get_feature('total_length', NEURON, neurite_type=NeuriteType.basal_dendrite)
    assert len(tl) == 1
    assert_allclose(tl, (418.43241644,))


def test_total_length_apical():
    tl = get_feature('total_length', NEURON, neurite_type=NeuriteType.apical_dendrite)
    assert len(tl) == 1
    assert_allclose(tl, (214.37304578,))


def test_section_lengths_invalid():
    s = get_feature('section_lengths', NEURON, neurite_type=NeuriteType.soma)
    assert len(s) == 0
    s = get_feature('section_lengths', NEURON, neurite_type=NeuriteType.undefined)
    assert len(s) == 0


def test_section_path_distances_axon():
    path_lengths = get_feature('section_path_distances', NEURON, neurite_type=NeuriteType.axon)
    assert len(path_lengths) == 21


def test_segment_lengths():
    ref_seglen = nf.segment_lengths(NEURON)
    seglen = get_feature('segment_lengths', NEURON)
    assert len(seglen) == 840
    assert_allclose(seglen, ref_seglen)

    seglen = get_feature('segment_lengths', NEURON, neurite_type=NeuriteType.all)
    assert len(seglen) == 840
    assert_allclose(seglen, ref_seglen)


def test_local_bifurcation_angles():
    ref_local_bifangles = list(nf.local_bifurcation_angles(NEURON))

    local_bifangles = get_feature('local_bifurcation_angles', NEURON)
    assert len(local_bifangles) == 40
    assert_allclose(local_bifangles, ref_local_bifangles)
    local_bifangles = get_feature('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.all)
    assert len(local_bifangles) == 40
    assert_allclose(local_bifangles, ref_local_bifangles)

    s = get_feature('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.axon)
    assert len(s) == 10

    s = get_feature('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.basal_dendrite)
    assert len(s) == 20

    s = get_feature('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.apical_dendrite)
    assert len(s) == 10


def test_local_bifurcation_angles_invalid():
    s = get_feature('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.soma)
    assert len(s) == 0
    s = get_feature('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.undefined)
    assert len(s) == 0


def test_remote_bifurcation_angles():
    ref_remote_bifangles = list(nf.remote_bifurcation_angles(NEURON))
    remote_bifangles = get_feature('remote_bifurcation_angles', NEURON)
    assert len(remote_bifangles) == 40
    assert_allclose(remote_bifangles, ref_remote_bifangles)
    remote_bifangles = get_feature('remote_bifurcation_angles',
                                   NEURON, neurite_type=NeuriteType.all)
    assert len(remote_bifangles) == 40
    assert_allclose(remote_bifangles, ref_remote_bifangles)

    s = get_feature('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.axon)
    assert len(s) == 10

    s = get_feature('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.basal_dendrite)
    assert len(s) == 20

    s = get_feature('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.apical_dendrite)
    assert len(s) == 10


def test_remote_bifurcation_angles_invalid():
    s = get_feature('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.soma)
    assert len(s) == 0
    s = get_feature('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.undefined)
    assert len(s) == 0


def test_segment_radial_distances_origin():
    origin = (-100, -200, -300)
    ref_segs = nf.segment_radial_distances(NEURON)
    ref_segs_origin = nf.segment_radial_distances(NEURON, origin=origin)

    rad_dists = get_feature('segment_radial_distances', NEURON)
    rad_dists_origin = get_feature('segment_radial_distances', NEURON, origin=origin)

    assert np.all(rad_dists == ref_segs)
    assert np.all(rad_dists_origin == ref_segs_origin)
    assert np.all(rad_dists_origin != ref_segs)

    nrns = [nm.load_neuron(Path(SWC_PATH, f)) for
            f in ('point_soma_single_neurite.swc', 'point_soma_single_neurite2.swc')]
    pop = Population(nrns)
    rad_dist_nrns = []
    for nrn in nrns:
        rad_dist_nrns.extend(nm.get('segment_radial_distances', nrn))

    rad_dist_nrns = np.array(rad_dist_nrns)
    rad_dist_pop = nm.get('segment_radial_distances', pop)
    assert_allclose(rad_dist_nrns, rad_dist_pop)


def test_section_radial_distances_endpoint():
    ref_sec_rad_dist = nf.section_radial_distances(NEURON)

    rad_dists = get_feature('section_radial_distances', NEURON)

    assert len(rad_dists) == 84
    assert np.all(rad_dists == ref_sec_rad_dist)

    nrns = [nm.load_neuron(Path(SWC_PATH, f)) for
            f in ('point_soma_single_neurite.swc', 'point_soma_single_neurite2.swc')]
    pop = Population(nrns)
    rad_dist_nrns = [nm.get('section_radial_distances', nrn) for nrn in nrns]
    rad_dist_pop = nm.get('section_radial_distances', pop)
    assert_items_equal(rad_dist_pop, rad_dist_nrns)


def test_section_radial_distances_origin():
    origin = (-100, -200, -300)
    ref_sec_rad_dist_origin = nf.section_radial_distances(NEURON, origin=origin)
    rad_dists = get_feature('section_radial_distances', NEURON, origin=origin)
    assert len(rad_dists) == 84
    assert np.all(rad_dists == ref_sec_rad_dist_origin)


def test_section_radial_axon():
    rad_dists = get_feature('section_radial_distances', NEURON, neurite_type=NeuriteType.axon)
    assert len(rad_dists) == 21


def test_number_of_sections_all():
    assert get_feature('number_of_sections', NEURON)[0] == 84
    assert get_feature('number_of_sections', NEURON, neurite_type=NeuriteType.all)[0] == 84


def test_number_of_sections_axon():
    assert get_feature('number_of_sections', NEURON, neurite_type=NeuriteType.axon)[0] == 21


def test_number_of_sections_basal():
    assert get_feature('number_of_sections', NEURON, neurite_type=NeuriteType.basal_dendrite)[0] == 42


def test_n_sections_apical():
    assert get_feature('number_of_sections', NEURON, neurite_type=NeuriteType.apical_dendrite)[0] == 21


def test_section_number_invalid():
    assert get_feature('number_of_sections', NEURON, neurite_type=NeuriteType.soma)[0] == 0
    assert get_feature('number_of_sections', NEURON, neurite_type=NeuriteType.undefined)[0] == 0


def test_per_neurite_number_of_sections():
    nsecs = get_feature('number_of_sections_per_neurite', NEURON)
    assert len(nsecs) == 4
    assert np.all(nsecs == [21, 21, 21, 21])


def test_per_neurite_number_of_sections_axon():
    nsecs = get_feature('number_of_sections_per_neurite', NEURON, neurite_type=NeuriteType.axon)
    assert len(nsecs) == 1
    assert nsecs == [21]


def test_n_sections_per_neurite_basal():
    nsecs = get_feature('number_of_sections_per_neurite', NEURON,
                        neurite_type=NeuriteType.basal_dendrite)
    assert len(nsecs) == 2
    assert np.all(nsecs == [21, 21])


def test_n_sections_per_neurite_apical():
    nsecs = get_feature('number_of_sections_per_neurite', NEURON,
                        neurite_type=NeuriteType.apical_dendrite)
    assert len(nsecs) == 1
    assert np.all(nsecs == [21])


def test_neurite_number():
    assert get_feature('number_of_neurites', NEURON)[0] == 4
    assert get_feature('number_of_neurites', NEURON, neurite_type=NeuriteType.all)[0] == 4
    assert get_feature('number_of_neurites', NEURON, neurite_type=NeuriteType.axon)[0] == 1
    assert get_feature('number_of_neurites', NEURON, neurite_type=NeuriteType.basal_dendrite)[0] == 2
    assert get_feature('number_of_neurites', NEURON, neurite_type=NeuriteType.apical_dendrite)[0] == 1
    assert get_feature('number_of_neurites', NEURON, neurite_type=NeuriteType.soma)[0] == 0
    assert get_feature('number_of_neurites', NEURON, neurite_type=NeuriteType.undefined)[0] == 0


def test_trunk_origin_radii():
    assert_allclose(get_feature('trunk_origin_radii', NEURON),
                    [0.85351288499400002,
                     0.18391483031299999,
                     0.66943255462899998,
                     0.14656092843999999])

    assert_allclose(get_feature('trunk_origin_radii', NEURON, neurite_type=NeuriteType.apical_dendrite),
                    [0.14656092843999999])
    assert_allclose(get_feature('trunk_origin_radii', NEURON, neurite_type=NeuriteType.basal_dendrite),
                    [0.18391483031299999,
                     0.66943255462899998])
    assert_allclose(get_feature('trunk_origin_radii', NEURON, neurite_type=NeuriteType.axon),
                    [0.85351288499400002])


def test_get_trunk_section_lengths():
    assert_allclose(get_feature('trunk_section_lengths', NEURON),
                    [9.579117366740002,
                     7.972322416776259,
                     8.2245287740603779,
                     9.212707985134525])
    assert_allclose(get_feature('trunk_section_lengths', NEURON, neurite_type=NeuriteType.apical_dendrite),
                    [9.212707985134525])
    assert_allclose(get_feature('trunk_section_lengths', NEURON, neurite_type=NeuriteType.basal_dendrite),
                    [7.972322416776259, 8.2245287740603779])
    assert_allclose(get_feature('trunk_section_lengths', NEURON, neurite_type=NeuriteType.axon),
                    [9.579117366740002])


def test_soma_radii():
    assert_allclose(get_feature('soma_radii', NEURON)[0], 0.13065629648763766)


def test_soma_surface_areas():
    area = 4. * math.pi * get_feature('soma_radii', NEURON)[0] ** 2
    assert_allclose(get_feature('soma_surface_areas', NEURON), area)


def test_sholl_frequency():
    assert_allclose(get_feature('sholl_frequency', NEURON),
                    [4, 8, 8, 14, 9, 8, 7, 7])

    assert_allclose(get_feature('sholl_frequency', NEURON, neurite_type=NeuriteType.all),
                    [4, 8, 8, 14, 9, 8, 7, 7])

    assert_allclose(get_feature('sholl_frequency', NEURON, neurite_type=NeuriteType.apical_dendrite),
                    [1, 2, 2, 2, 2, 2, 1, 1])

    assert_allclose(get_feature('sholl_frequency', NEURON, neurite_type=NeuriteType.basal_dendrite),
                    [2, 4, 4, 6, 5, 4, 4, 4])

    assert_allclose(get_feature('sholl_frequency', NEURON, neurite_type=NeuriteType.axon),
                    [1, 2, 2, 6, 2, 2, 2, 2])


@pytest.mark.skip('test_get_segment_lengths is disabled in test_get_features')
def test_section_path_distances_endpoint():

    ref_sec_path_len_start = list(iter_neurites(NEURON, sec.start_point_path_length))
    ref_sec_path_len = list(iter_neurites(NEURON, sec.end_point_path_length))
    path_lengths = get_feature('section_path_distances', NEURON)
    assert ref_sec_path_len != ref_sec_path_len_start
    assert len(path_lengths) == 84
    assert np.all(path_lengths == ref_sec_path_len)


@pytest.mark.skip('test_get_segment_lengths is disabled in test_get_features')
def test_section_path_distances_start_point():
    ref_sec_path_len_start = list(iter_neurites(NEURON, sec.start_point_path_length))
    path_lengths = get_feature('section_path_distances', NEURON, use_start_point=True)
    assert len(path_lengths) == 84
    assert np.all(path_lengths == ref_sec_path_len_start)


def test_partition():
    assert np.all(get_feature('partition', NRNS)[:10] == np.array(
        [19.,  17.,  15.,  13.,  11.,   9.,   7.,   5.,   3.,   1.]))


def test_partition_asymmetry():
    assert_allclose(get_feature('partition_asymmetry', NRNS)[:10], np.array([0.9, 0.88888889, 0.875,
                                                                             0.85714286, 0.83333333,

                                                                             0.8, 0.75,  0.66666667,
                                                                             0.5,  0.]))


def test_partition_asymmetry_length():
    assert_allclose(get_feature('partition_asymmetry_length', NRNS)[:1], np.array([0.853925]))


def test_section_strahler_orders():
    path = Path(SWC_PATH, 'strahler.swc')
    n = nm.load_neuron(path)
    assert_allclose(get_feature('section_strahler_orders', n),
                    [4, 1, 4, 3, 2, 1, 1, 2, 1, 1, 3, 1, 3, 2, 1, 1, 2, 1, 1])
