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

"""Test ``neurom.features.get`` function."""
import itertools
import math
from io import StringIO
from pathlib import Path

import neurom as nm
import numpy as np
from neurom import features, iter_sections, load_morphology, load_morphologies
from neurom.core.population import Population
from neurom.core.types import NeuriteType
from neurom.exceptions import NeuroMError
from neurom.features import neurite, NameSpace

import pytest
from numpy import testing as npt
from numpy.testing import assert_allclose

DATA_PATH = Path(__file__).parent.parent / 'data'
NRN_FILES = [DATA_PATH / 'h5/v1' / f
             for f in ('Neuron.h5', 'Neuron_2_branch.h5', 'bio_neuron-001.h5')]
POP = load_morphologies(NRN_FILES)
NRN = POP[0]

SWC_PATH = DATA_PATH / 'swc'
NEURON_PATH = SWC_PATH / 'Neuron.swc'
NEURON = load_morphology(NEURON_PATH)
NEURITES = (NeuriteType.axon,
            NeuriteType.apical_dendrite,
            NeuriteType.basal_dendrite,
            NeuriteType.all)


def _stats(seq):
    seq = list(itertools.chain(*seq)) if isinstance(seq[0], list) else seq
    return np.min(seq), np.max(seq), np.sum(seq), np.mean(seq)


def test_get_raises():
    with pytest.raises(NeuroMError, match="Only Neurite, Morphology, Population or list, tuple of Neurite, Morphology"):
        features.get('soma_radius', (n for n in POP))
    with pytest.raises(NeuroMError, match="Cant apply 'invalid' feature"):
        features.get('invalid', NRN)


def test_register_existing_feature():
    with pytest.raises(NeuroMError):
        features._register_feature(NameSpace.NEURITE, 'total_area', lambda n: None, ())
    with pytest.raises(NeuroMError):
        features._register_feature(NameSpace.NEURON, 'total_length_per_neurite', lambda n: None, ())
    with pytest.raises(NeuroMError):
        features._register_feature(NameSpace.POPULATION, 'sholl_frequency', lambda n: None, ())


def test_number_of_sections():
    assert features.get('number_of_sections', POP) == [84, 42, 202]
    assert features.get('number_of_sections', POP,
                        neurite_type=NeuriteType.all) == [84, 42, 202]
    assert features.get('number_of_sections', POP,
                        neurite_type=NeuriteType.axon) == [21, 21, 179]
    assert features.get('number_of_sections', POP,
                        neurite_type=NeuriteType.apical_dendrite) == [21, 0, 0]
    assert features.get('number_of_sections', POP,
                        neurite_type=NeuriteType.basal_dendrite) == [42, 21, 23]

    assert features.get('number_of_sections', NEURON) == 84
    assert features.get('number_of_sections', NEURON,
                        neurite_type=NeuriteType.all) == 84
    assert features.get('number_of_sections', NEURON,
                        neurite_type=NeuriteType.axon) == 21
    assert features.get('number_of_sections', NEURON,
                        neurite_type=NeuriteType.basal_dendrite) == 42
    assert features.get('number_of_sections', NEURON,
                        neurite_type=NeuriteType.apical_dendrite) == 21

    assert features.get('number_of_sections', NEURON.neurites) == [21, 21, 21, 21]
    assert features.get('number_of_sections', NEURON.neurites[0]) == 21

    assert features.get('number_of_sections', NEURON, neurite_type=NeuriteType.soma) == 0
    assert features.get('number_of_sections', NEURON, neurite_type=NeuriteType.undefined) == 0


def test_max_radial_distance():
    assert_allclose(
        features.get('max_radial_distance', POP),
        [99.58945832, 94.43342439, 1053.77939245])
    assert_allclose(
        features.get('max_radial_distance', POP, neurite_type=NeuriteType.all),
        [99.58945832, 94.43342439, 1053.77939245])
    assert_allclose(
        features.get('max_radial_distance', POP, neurite_type=NeuriteType.axon),
        [82.442545, 82.442545, 1053.779392])
    assert_allclose(
        features.get('max_radial_distance', POP, neurite_type=NeuriteType.basal_dendrite),
        [94.43342563, 94.43342439, 207.56977859])

    assert_allclose(
        features.get('max_radial_distance', NRN), 99.58945832)
    assert_allclose(
        features.get('max_radial_distance', NRN, neurite_type=NeuriteType.all), 99.58945832)
    assert_allclose(features.get(
        'max_radial_distance', NRN, neurite_type=NeuriteType.apical_dendrite), 99.589458)

    assert_allclose(
        features.get('max_radial_distance', NRN.neurites),
        [99.58946, 80.05163, 94.433426, 82.44254])
    assert_allclose(
        features.get('max_radial_distance', NRN.neurites[0]), 99.58946)


def test_section_tortuosity():
    assert_allclose(
        _stats(features.get('section_tortuosity', POP)),
        (1.0, 4.657, 440.408, 1.342), rtol=1e-3)
    assert_allclose(
        _stats(features.get('section_tortuosity', POP, neurite_type=NeuriteType.all)),
        (1.0, 4.657, 440.408, 1.342), rtol=1e-3)
    assert_allclose(
        _stats(features.get('section_tortuosity', POP, neurite_type=NeuriteType.apical_dendrite)),
        (1.070, 1.573, 26.919, 1.281), rtol=1e-3)
    assert_allclose(
        _stats(features.get('section_tortuosity', POP, neurite_type=NeuriteType.basal_dendrite)),
        (1.042, 1.674, 106.596, 1.239), rtol=1e-3)

    assert_allclose(
        _stats(features.get('section_tortuosity', NRN)),
        (1.070, 1.573, 106.424, 1.266), rtol=1e-3)
    assert_allclose(
        _stats(features.get('section_tortuosity', NRN, neurite_type=NeuriteType.all)),
        (1.070, 1.573, 106.424, 1.266), rtol=1e-3)
    assert_allclose(
        _stats(features.get('section_tortuosity', NRN, neurite_type=NeuriteType.apical_dendrite)),
        (1.070, 1.573, 26.919, 1.281), rtol=1e-3)
    assert_allclose(
        _stats(features.get('section_tortuosity', NRN, neurite_type=NeuriteType.basal_dendrite)),
        (1.078, 1.550, 51.540, 1.227), rtol=1e-3)


def test_number_of_segments():
    assert features.get('number_of_segments', POP) == [840, 419, 5179]
    assert features.get('number_of_segments', POP,
                        neurite_type=NeuriteType.all) == [840, 419, 5179]
    assert features.get('number_of_segments', POP,
                        neurite_type=NeuriteType.axon) == [210, 209, 4508]
    assert features.get('number_of_segments', POP,
                        neurite_type=NeuriteType.apical_dendrite) == [210, 0, 0]
    assert features.get('number_of_segments', POP,
                        neurite_type=NeuriteType.basal_dendrite) == [420, 210, 671]

    assert features.get('number_of_segments', NRN) == 840
    assert features.get('number_of_segments', NRN,
                        neurite_type=NeuriteType.all) == 840
    assert features.get('number_of_segments', NRN,
                        neurite_type=NeuriteType.axon) == 210
    assert features.get('number_of_segments', NRN,
                        neurite_type=NeuriteType.apical_dendrite) == 210
    assert features.get('number_of_segments', NRN,
                        neurite_type=NeuriteType.basal_dendrite) == 420


def test_number_of_neurites():
    assert features.get('number_of_neurites', POP) == [4, 2, 4]
    assert features.get('number_of_neurites', POP,
                        neurite_type=NeuriteType.all) == [4, 2, 4]
    assert features.get('number_of_neurites', POP,
                        neurite_type=NeuriteType.axon) == [1, 1, 1]
    assert features.get('number_of_neurites', POP,
                        neurite_type=NeuriteType.apical_dendrite) == [1, 0, 0]
    assert features.get('number_of_neurites', POP,
                        neurite_type=NeuriteType.basal_dendrite) == [2, 1, 3]

    assert features.get('number_of_neurites', NRN) == 4
    assert features.get('number_of_neurites', NRN,
                        neurite_type=NeuriteType.all) == 4
    assert features.get('number_of_neurites', NRN,
                        neurite_type=NeuriteType.axon) == 1
    assert features.get('number_of_neurites', NRN,
                        neurite_type=NeuriteType.apical_dendrite) == 1
    assert features.get('number_of_neurites', NRN,
                        neurite_type=NeuriteType.basal_dendrite) == 2


def test_number_of_bifurcations():
    assert features.get('number_of_bifurcations', POP) == [40, 20, 97]
    assert features.get('number_of_bifurcations', POP,
                        neurite_type=NeuriteType.all) == [40, 20, 97]
    assert features.get('number_of_bifurcations', POP,
                        neurite_type=NeuriteType.axon) == [10, 10, 87]
    assert features.get('number_of_bifurcations', POP,
                        neurite_type=NeuriteType.apical_dendrite) == [10, 0, 0]
    assert features.get('number_of_bifurcations', POP,
                        neurite_type=NeuriteType.basal_dendrite) == [20, 10, 10]

    assert features.get('number_of_bifurcations', NRN) == 40
    assert features.get('number_of_bifurcations', NRN,
                        neurite_type=NeuriteType.all) == 40
    assert features.get('number_of_bifurcations', NRN,
                        neurite_type=NeuriteType.axon) == 10
    assert features.get('number_of_bifurcations', NRN,
                        neurite_type=NeuriteType.apical_dendrite) == 10
    assert features.get('number_of_bifurcations', NRN,
                        neurite_type=NeuriteType.basal_dendrite) == 20


def test_number_of_forking_points():
    assert features.get('number_of_forking_points', POP) == [40, 20, 98]
    assert features.get('number_of_forking_points', POP,
                        neurite_type=NeuriteType.all) == [40, 20, 98]
    assert features.get('number_of_forking_points', POP,
                        neurite_type=NeuriteType.axon) == [10, 10, 88]
    assert features.get('number_of_forking_points', POP,
                        neurite_type=NeuriteType.apical_dendrite) == [10, 0, 0]
    assert features.get('number_of_forking_points', POP,
                        neurite_type=NeuriteType.basal_dendrite) == [20, 10, 10]

    assert features.get('number_of_forking_points', NRN) == 40
    assert features.get('number_of_forking_points', NRN,
                        neurite_type=NeuriteType.all) == 40
    assert features.get('number_of_forking_points', NRN,
                        neurite_type=NeuriteType.axon) == 10
    assert features.get('number_of_forking_points', NRN,
                        neurite_type=NeuriteType.apical_dendrite) == 10
    assert features.get('number_of_forking_points', NRN,
                        neurite_type=NeuriteType.basal_dendrite) == 20


def test_number_of_leaves():
    assert features.get('number_of_leaves', POP) == [44, 22, 103]
    assert features.get('number_of_leaves', POP,
                        neurite_type=NeuriteType.all) == [44, 22, 103]
    assert features.get('number_of_leaves', POP,
                        neurite_type=NeuriteType.axon) == [11, 11, 90]
    assert features.get('number_of_leaves', POP,
                        neurite_type=NeuriteType.apical_dendrite) == [11, 0, 0]
    assert features.get('number_of_leaves', POP,
                        neurite_type=NeuriteType.basal_dendrite) == [22, 11, 13]

    assert features.get('number_of_leaves', NRN) == 44
    assert features.get('number_of_leaves', NRN,
                        neurite_type=NeuriteType.all) == 44
    assert features.get('number_of_leaves', NRN,
                        neurite_type=NeuriteType.axon) == 11
    assert features.get('number_of_leaves', NRN,
                        neurite_type=NeuriteType.apical_dendrite) == 11
    assert features.get('number_of_leaves', NRN,
                        neurite_type=NeuriteType.basal_dendrite) == 22


def test_total_length():
    assert_allclose(
        features.get('total_length', POP),
        [840.68522362011538, 418.83424432013902, 13250.825773939932])
    assert_allclose(
        features.get('total_length', POP, neurite_type=NeuriteType.all),
        [840.68522362011538, 418.83424432013902, 13250.825773939932])
    assert_allclose(
        features.get('total_length', POP, neurite_type=NeuriteType.axon),
        [207.8797736031714, 207.81088341560977, 11767.156115224638])
    assert_allclose(
        features.get('total_length', POP, neurite_type=NeuriteType.apical_dendrite),
        [214.37302709169489, 0, 0])
    assert_allclose(
        features.get('total_length', POP, neurite_type=NeuriteType.basal_dendrite),
        [418.43242292524889, 211.02336090452931, 1483.6696587152967])

    assert_allclose(
        features.get('total_length', NEURON, neurite_type=NeuriteType.axon),
        207.87975221)
    assert_allclose(
        features.get('total_length', NEURON, neurite_type=NeuriteType.basal_dendrite),
        418.432424)
    assert_allclose(
        features.get('total_length', NEURON, neurite_type=NeuriteType.apical_dendrite),
        214.37304578)
    assert_allclose(
        features.get('total_length', NEURON, neurite_type=NeuriteType.axon),
        207.87975221)
    assert_allclose(
        features.get('total_length', NEURON, neurite_type=NeuriteType.basal_dendrite),
        418.43241644)
    assert_allclose(
        features.get('total_length', NEURON, neurite_type=NeuriteType.apical_dendrite),
        214.37304578)


def test_trunk_angles():
    trunk_angles_pop = features.get('trunk_angles', POP, neurite_type=NeuriteType.basal_dendrite)
    trunk_angles_morphs = features.get(
        'trunk_angles',
        [i for i in POP],
        neurite_type=NeuriteType.basal_dendrite
    )
    trunk_angles_morphs_2 = np.concatenate([
        features.get('trunk_angles', i, neurite_type=NeuriteType.basal_dendrite)
        for i in POP
    ]).tolist()

    assert trunk_angles_pop == trunk_angles_morphs == trunk_angles_morphs_2


def test_neurite_lengths():
    actual = features.get('total_length_per_neurite', POP, neurite_type=NeuriteType.basal_dendrite)
    expected = [207.31504917144775, 211.11737489700317, 211.02336168289185,
                501.28893661499023, 133.21348762512207, 849.1672043800354]
    for a,e in zip(actual, expected):
        assert_allclose(a, e)

    assert_allclose(
        features.get('total_length_per_neurite', NEURON, neurite_type=NeuriteType.axon),
        (207.87975221,))
    assert_allclose(
        features.get('total_length_per_neurite', NEURON, neurite_type=NeuriteType.basal_dendrite),
        (211.11737442, 207.31504202))
    assert_allclose(
        features.get('total_length_per_neurite', NEURON, neurite_type=NeuriteType.apical_dendrite),
        (214.37304578,))


def test_segment_radii():
    assert_allclose(
        _stats(features.get('segment_radii', POP)),
        (0.079999998211860657, 1.2150000333786011, 1301.9191725363567, 0.20222416473071708))
    assert_allclose(
        _stats(features.get('segment_radii', POP, neurite_type=NeuriteType.all)),
        (0.079999998211860657, 1.2150000333786011, 1301.9191725363567, 0.20222416473071708))
    assert_allclose(
        _stats(features.get('segment_radii', POP, neurite_type=NeuriteType.apical_dendrite)),
        (0.13142434507608414, 1.0343990325927734, 123.41135908663273, 0.58767313850777492))
    assert_allclose(
        _stats(features.get('segment_radii', POP, neurite_type=NeuriteType.basal_dendrite)),
        (0.079999998211860657, 1.2150000333786011, 547.43900821779164, 0.42078324997524336))

    assert_allclose(
        _stats(features.get('segment_radii', NRN)),
        (0.12087134271860123, 1.0343990325927734, 507.01994501426816, 0.60359517263603357))
    assert_allclose(
        _stats(features.get('segment_radii', NRN, neurite_type=NeuriteType.all)),
        (0.12087134271860123, 1.0343990325927734, 507.01994501426816, 0.60359517263603357))
    assert_allclose(
        _stats(features.get('segment_radii', NRN, neurite_type=NeuriteType.apical_dendrite)),
        (0.13142434507608414, 1.0343990325927734, 123.41135908663273, 0.58767313850777492))
    assert_allclose(
        _stats(features.get('segment_radii', NRN, neurite_type=NeuriteType.basal_dendrite)),
        (0.14712842553853989, 1.0215770602226257, 256.71241207793355, 0.61122002875698467))


def test_segment_meander_angles():
    assert_allclose(
        _stats(features.get('segment_meander_angles', POP)),
        (0.0, 3.1415, 14637.9776, 2.3957), rtol=1e-3)
    assert_allclose(
        _stats(features.get('segment_meander_angles', POP, neurite_type=NeuriteType.all)),
        (0.0, 3.1415, 14637.9776, 2.3957), rtol=1e-3)
    assert_allclose(
        _stats(features.get('segment_meander_angles', POP, neurite_type=NeuriteType.apical_dendrite)),
        (0.3261, 3.0939, 461.9816, 2.4443), rtol=1e-4)
    assert_allclose(
        _stats(features.get('segment_meander_angles', POP, neurite_type=NeuriteType.basal_dendrite)),
        (0.0, 3.1415, 2926.2411, 2.4084), rtol=1e-4)

    assert_allclose(
        _stats(features.get('segment_meander_angles', NRN)),
        (0.32610, 3.12996, 1842.35, 2.43697), rtol=1e-5)
    assert_allclose(
        _stats(features.get('segment_meander_angles', NRN, neurite_type=NeuriteType.all)),
        (0.32610, 3.12996, 1842.35, 2.43697), rtol=1e-5)
    assert_allclose(
        _stats(features.get('segment_meander_angles', NRN, neurite_type=NeuriteType.apical_dendrite)),
        (0.32610, 3.09392, 461.981, 2.44434), rtol=1e-5)
    assert_allclose(
        _stats(features.get('segment_meander_angles', NRN, neurite_type=NeuriteType.basal_dendrite)),
        (0.47318, 3.12996, 926.338, 2.45063), rtol=1e-4)


def test_segment_meander_angles_single_section():
    m = nm.load_morphology(StringIO(u"""((CellBody) (-1 0 0 2) (1 0 0 2))
                                      ((Dendrite)
                                       (0 0 0 2)
                                       (1 0 0 2)
                                       (1 1 0 2)
                                       (2 1 0 2)
                                       (2 2 0 2)))"""), reader='asc')
    nrt = m.neurites[0]
    pop = [m]

    ref = [math.pi / 2, math.pi / 2, math.pi / 2]

    assert ref == features.get('segment_meander_angles', nrt)
    assert ref == features.get('segment_meander_angles', m)
    assert ref == features.get('segment_meander_angles', pop)


def test_neurite_volumes():
    assert_allclose(
        _stats(features.get('total_volume_per_neurite', POP)),
        (28.356406629821159, 281.24754646913954, 2249.4613918388391, 224.9461391838839))
    assert_allclose(
        _stats(features.get('total_volume_per_neurite', POP, neurite_type=NeuriteType.all)),
        (28.356406629821159, 281.24754646913954, 2249.4613918388391, 224.9461391838839))
    assert_allclose(
        _stats(features.get('total_volume_per_neurite', POP, neurite_type=NeuriteType.axon)),
        (276.58135508666612, 277.5357232437392, 830.85568094763551, 276.95189364921185))
    assert_allclose(
        _stats(features.get('total_volume_per_neurite', POP, neurite_type=NeuriteType.apical_dendrite)),
        (271.94122143951864, 271.94122143951864, 271.94122143951864, 271.94122143951864))
    assert_allclose(
        _stats(features.get('total_volume_per_neurite', POP, neurite_type=NeuriteType.basal_dendrite)),
        (28.356406629821159, 281.24754646913954, 1146.6644894516851, 191.1107482419475))

    assert_allclose(
        _stats(features.get('total_volume_per_neurite', NRN)),
        (271.9412, 281.2475, 1104.907, 276.2269), rtol=1e-5)
    assert_allclose(
        _stats(features.get('total_volume_per_neurite', NRN, neurite_type=NeuriteType.all)),
        (271.9412, 281.2475, 1104.907, 276.2269), rtol=1e-5)
    assert_allclose(
        _stats(features.get('total_volume_per_neurite', NRN, neurite_type=NeuriteType.axon)),
        (276.7386, 276.7386, 276.7386, 276.7386), rtol=1e-5)
    assert_allclose(
        _stats(features.get('total_volume_per_neurite', NRN, neurite_type=NeuriteType.apical_dendrite)),
        (271.9412, 271.9412, 271.9412, 271.9412), rtol=1e-5)
    assert_allclose(
        _stats(features.get('total_volume_per_neurite', NRN, neurite_type=NeuriteType.basal_dendrite)),
        (274.9803, 281.2475, 556.2279, 278.1139), rtol=1e-5)


def test_neurite_density():
    assert_allclose(
        _stats(features.get('neurite_volume_density', POP)),
        (6.1847539631150784e-06, 0.52464681266899216, 1.9767794901940539, 0.19767794901940539))
    assert_allclose(
        _stats(features.get('neurite_volume_density', POP, neurite_type=NeuriteType.all)),
        (6.1847539631150784e-06, 0.52464681266899216, 1.9767794901940539, 0.19767794901940539))
    assert_allclose(
        _stats(features.get('neurite_volume_density', POP, neurite_type=NeuriteType.axon)),
        (6.1847539631150784e-06, 0.26465213325053372, 0.5275513670655404, 0.1758504556885134), 1e-6)
    assert_allclose(
        _stats(features.get('neurite_volume_density', POP, neurite_type=NeuriteType.apical_dendrite)),
        (0.43756606998299519, 0.43756606998299519, 0.43756606998299519, 0.43756606998299519))
    assert_allclose(
        _stats(features.get('neurite_volume_density', POP, neurite_type=NeuriteType.basal_dendrite)),
        (0.00034968816544949771, 0.52464681266899216, 1.0116620531455183, 0.16861034219091972))

    assert_allclose(
        _stats(features.get('neurite_volume_density', NRN)),
        (0.24068543213643726, 0.52464681266899216, 1.4657913638494682, 0.36644784096236704))
    assert_allclose(
        _stats(features.get('neurite_volume_density', NRN, neurite_type=NeuriteType.all)),
        (0.24068543213643726, 0.52464681266899216, 1.4657913638494682, 0.36644784096236704))
    assert_allclose(
        _stats(features.get('neurite_volume_density', NRN, neurite_type=NeuriteType.axon)),
        (0.26289304906104355, 0.26289304906104355, 0.26289304906104355, 0.26289304906104355))
    assert_allclose(
        _stats(features.get('neurite_volume_density', NRN, neurite_type=NeuriteType.apical_dendrite)),
        (0.43756606998299519, 0.43756606998299519, 0.43756606998299519, 0.43756606998299519))
    assert_allclose(
        _stats(features.get('neurite_volume_density', NRN, neurite_type=NeuriteType.basal_dendrite)),
        (0.24068543213643726, 0.52464681266899216, 0.76533224480542938, 0.38266612240271469))


def test_morphology_volume_density():

    volume_density = features.get("volume_density", NEURON)

    # volume density should not be calculated as the sum of the neurite volume densities,
    # because it is not additive
    volume_density_from_neurites = sum(
        features.get("volume_density", neu) for neu in NEURON.neurites
    )

    # calculating the convex hull per neurite results into smaller hull volumes and higher
    # neurite_volume / hull_volume ratios
    assert not np.isclose(volume_density, volume_density_from_neurites)
    assert volume_density < volume_density_from_neurites


def test_section_lengths():
    ref_seclen = [n.length for n in iter_sections(NEURON)]
    seclen = features.get('section_lengths', NEURON)
    assert len(seclen) == 84
    assert_allclose(seclen, ref_seclen)

    s = features.get('section_lengths', NEURON, neurite_type=NeuriteType.axon)
    assert len(s) == 21
    s = features.get('section_lengths', NEURON, neurite_type=NeuriteType.basal_dendrite)
    assert len(s) == 42
    s = features.get('section_lengths', NEURON, neurite_type=NeuriteType.apical_dendrite)
    assert len(s) == 21

    s = features.get('section_lengths', NEURON, neurite_type=NeuriteType.soma)
    assert len(s) == 0
    s = features.get('section_lengths', NEURON, neurite_type=NeuriteType.undefined)
    assert len(s) == 0


def test_section_path_distances():
    path_distances = features.get('section_path_distances', POP)
    assert len(path_distances) == 328
    assert sum(len(features.get('section_path_distances', m)) for m in POP) == 328

    path_lengths = features.get('section_path_distances', NEURON, neurite_type=NeuriteType.axon)
    assert len(path_lengths) == 21


def test_segment_lengths():
    ref_seglen = np.concatenate([neurite.segment_lengths(s) for s in NEURON.neurites])
    seglen = features.get('segment_lengths', NEURON)
    assert len(seglen) == 840
    assert_allclose(seglen, ref_seglen)

    seglen = features.get('segment_lengths', NEURON, neurite_type=NeuriteType.all)
    assert len(seglen) == 840
    assert_allclose(seglen, ref_seglen)


def test_local_bifurcation_angles():
    ref_local_bifangles = np.concatenate([neurite.local_bifurcation_angles(s)
                                          for s in NEURON.neurites])

    local_bifangles = features.get('local_bifurcation_angles', NEURON)
    assert len(local_bifangles) == 40
    assert_allclose(local_bifangles, ref_local_bifangles)
    local_bifangles = features.get('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.all)
    assert len(local_bifangles) == 40
    assert_allclose(local_bifangles, ref_local_bifangles)

    s = features.get('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.axon)
    assert len(s) == 10
    s = features.get('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.basal_dendrite)
    assert len(s) == 20
    s = features.get('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.apical_dendrite)
    assert len(s) == 10

    s = features.get('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.soma)
    assert len(s) == 0
    s = features.get('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.undefined)
    assert len(s) == 0


def test_remote_bifurcation_angles():
    ref_remote_bifangles = np.concatenate([neurite.remote_bifurcation_angles(s)
                                           for s in NEURON.neurites])
    remote_bifangles = features.get('remote_bifurcation_angles', NEURON)
    assert len(remote_bifangles) == 40
    assert_allclose(remote_bifangles, ref_remote_bifangles)
    remote_bifangles = features.get('remote_bifurcation_angles',
                                    NEURON, neurite_type=NeuriteType.all)
    assert len(remote_bifangles) == 40
    assert_allclose(remote_bifangles, ref_remote_bifangles)

    s = features.get('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.axon)
    assert len(s) == 10
    s = features.get('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.basal_dendrite)
    assert len(s) == 20
    s = features.get('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.apical_dendrite)
    assert len(s) == 10

    s = features.get('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.soma)
    assert len(s) == 0
    s = features.get('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.undefined)
    assert len(s) == 0


def test_segment_radial_distances_origin():
    origin = (-100, -200, -300)
    ref_segs = np.concatenate([neurite.segment_radial_distances(s) for s in NEURON.neurites])
    ref_segs_origin = np.concatenate([neurite.segment_radial_distances(s, origin)
                                      for s in NEURON.neurites])

    rad_dists = features.get('segment_radial_distances', NEURON)
    rad_dists_origin = features.get('segment_radial_distances', NEURON, origin=origin)

    assert np.all(rad_dists == ref_segs)
    assert np.all(rad_dists_origin == ref_segs_origin)
    assert np.all(rad_dists_origin != ref_segs)

    morphs = [nm.load_morphology(Path(SWC_PATH, f)) for
            f in ('point_soma_single_neurite.swc', 'point_soma_single_neurite2.swc')]
    pop = Population(morphs)
    rad_dist_morphs = []
    for m in morphs:
        rad_dist_morphs.extend(features.get('segment_radial_distances', m))

    rad_dist_morphs = np.array(rad_dist_morphs)
    rad_dist_pop = features.get('segment_radial_distances', pop)
    assert_allclose(rad_dist_morphs, rad_dist_pop)


def test_section_radial_distances_endpoint():
    ref_sec_rad_dist = np.concatenate([neurite.section_radial_distances(s)
                                       for s in NEURON.neurites])
    rad_dists = features.get('section_radial_distances', NEURON)

    assert len(rad_dists) == 84
    assert np.all(rad_dists == ref_sec_rad_dist)

    morphs = [nm.load_morphology(Path(SWC_PATH, f)) for
            f in ('point_soma_single_neurite.swc', 'point_soma_single_neurite2.swc')]
    pop = Population(morphs)
    rad_dist_morphs = [v for m in morphs for v in features.get('section_radial_distances', m)]
    rad_dist_pop = features.get('section_radial_distances', pop)
    assert_allclose(rad_dist_pop, rad_dist_morphs)

    rad_dists = features.get('section_radial_distances', NEURON, neurite_type=NeuriteType.axon)
    assert len(rad_dists) == 21


def test_section_radial_distances_origin():
    origin = (-100, -200, -300)
    ref_sec_rad_dist_origin = np.concatenate([neurite.section_radial_distances(s, origin)
                                              for s in NEURON.neurites])
    rad_dists = features.get('section_radial_distances', NEURON, origin=origin)
    assert len(rad_dists) == 84
    assert np.all(rad_dists == ref_sec_rad_dist_origin)


def test_number_of_sections_per_neurite():
    nsecs = features.get('number_of_sections_per_neurite', NEURON)
    assert len(nsecs) == 4
    assert np.all(nsecs == [21, 21, 21, 21])

    nsecs = features.get('number_of_sections_per_neurite', NEURON, neurite_type=NeuriteType.axon)
    assert len(nsecs) == 1
    assert nsecs == [21]

    nsecs = features.get('number_of_sections_per_neurite', NEURON,
                         neurite_type=NeuriteType.basal_dendrite)
    assert len(nsecs) == 2
    assert np.all(nsecs == [21, 21])

    nsecs = features.get('number_of_sections_per_neurite', NEURON,
                         neurite_type=NeuriteType.apical_dendrite)
    assert len(nsecs) == 1
    assert np.all(nsecs == [21])


def test_trunk_origin_radii():
    assert_allclose(
        features.get('trunk_origin_radii', NEURON),
        [0.85351288499400002, 0.18391483031299999, 0.66943255462899998, 0.14656092843999999])
    assert_allclose(
        features.get('trunk_origin_radii', NEURON, neurite_type=NeuriteType.apical_dendrite),
        [0.14656092843999999])
    assert_allclose(
        features.get('trunk_origin_radii', NEURON, neurite_type=NeuriteType.basal_dendrite),
        [0.18391483031299999, 0.66943255462899998])
    assert_allclose(
        features.get('trunk_origin_radii', NEURON, neurite_type=NeuriteType.axon),
        [0.85351288499400002])


def test_trunk_section_lengths():
    assert_allclose(
        features.get('trunk_section_lengths', NEURON),
        [9.579117366740002, 7.972322416776259, 8.2245287740603779, 9.212707985134525])
    assert_allclose(
        features.get('trunk_section_lengths', NEURON, neurite_type=NeuriteType.apical_dendrite),
        [9.212707985134525])
    assert_allclose(
        features.get('trunk_section_lengths', NEURON, neurite_type=NeuriteType.basal_dendrite),
        [7.972322416776259, 8.2245287740603779])
    assert_allclose(
        features.get('trunk_section_lengths', NEURON, neurite_type=NeuriteType.axon),
        [9.579117366740002])


def test_soma_radius():
    assert_allclose(features.get('soma_radius', NEURON), 0.13065629648763766)


def test_soma_surface_area():
    area = 4. * math.pi * features.get('soma_radius', NEURON) ** 2
    assert_allclose(features.get('soma_surface_area', NEURON), area)


def test_sholl_frequency():
    assert_allclose(features.get('sholl_frequency', NEURON),
                    [4, 8, 8, 14, 9, 8, 7, 7, 7, 5])

    assert_allclose(features.get('sholl_frequency', NEURON, neurite_type=NeuriteType.all),
                    [4, 8, 8, 14, 9, 8, 7, 7, 7, 5])

    assert_allclose(
        features.get('sholl_frequency', NEURON, neurite_type=NeuriteType.apical_dendrite),
        [1, 2, 2, 2, 2, 2, 1, 1, 3, 3])

    assert_allclose(
        features.get('sholl_frequency', NEURON, neurite_type=NeuriteType.basal_dendrite),
        [2, 4, 4, 6, 5, 4, 4, 4, 2, 2])

    assert_allclose(features.get('sholl_frequency', NEURON, neurite_type=NeuriteType.axon),
                    [1, 2, 2, 6, 2, 2, 2, 2, 2])

    assert len(features.get('sholl_frequency', POP)) == 108


    # check that the soma is taken into account for calculating max radius and num bins
    m = nm.load_morphology(
        """
        1  1  -10  0  0    5.0 -1
        2  3    0  0  0    0.1  1
        3  3   10  0  0    0.1  2
        """, reader="swc",
    )

    assert features.get('sholl_frequency', m, step_size=5.0) == [0, 1, 1, 1]

    # check that if there is no neurite of a specific type, an empty list is returned
    assert features.get('sholl_frequency', m, neurite_type=NeuriteType.axon) == []


def test_bifurcation_partitions():
    assert_allclose(features.get('bifurcation_partitions', POP)[:10],
                    [19., 17., 15., 13., 11., 9., 7., 5., 3., 1.])


def test_partition_asymmetry():
    assert_allclose(
        features.get('partition_asymmetry', POP)[:10],
        [0.9, 0.88888889, 0.875, 0.85714286, 0.83333333, 0.8, 0.75, 0.66666667, 0.5, 0.])


def test_partition_asymmetry_length():
    assert_allclose(features.get('partition_asymmetry_length', POP)[:1], np.array([0.853925]))


def test_section_strahler_orders():
    path = Path(SWC_PATH, 'strahler.swc')
    n = nm.load_morphology(path)
    assert_allclose(features.get('section_strahler_orders', n),
                    [4, 1, 4, 3, 2, 1, 1, 2, 1, 1, 3, 1, 3, 2, 1, 1, 2, 1, 1])


def test_section_bif_radial_distances():
    trm_rads = features.get('section_bif_radial_distances', NRN, neurite_type=nm.AXON)
    assert_allclose(trm_rads,
                    [8.842008561870646,
                     16.7440421479104,
                     23.070306480850533,
                     30.181121708042546,
                     36.62766031035137,
                     43.967487830324885,
                     51.91971040624528,
                     59.427722328770955,
                     66.25222507299583,
                     74.05119754074926])


def test_section_term_radial_distances():
    trm_rads = features.get('section_term_radial_distances', NRN, neurite_type=nm.APICAL_DENDRITE)
    assert_allclose(trm_rads,
                    [16.22099879395879,
                     25.992977561564082,
                     33.31600613822663,
                     42.721314797308175,
                     52.379508081911546,
                     59.44327819128149,
                     67.07832724133213,
                     79.97743930553612,
                     87.10434825508366,
                     97.25246040544428,
                     99.58945832481642])


def test_principal_direction_extents():
    m = nm.load_morphology(SWC_PATH / 'simple.swc')
    principal_dir = features.get('principal_direction_extents', m)
    assert_allclose(principal_dir, [10.99514 , 10.997688])

    # test with a realistic morphology
    m = nm.load_morphology(DATA_PATH / 'h5/v1' / 'bio_neuron-000.h5')

    assert_allclose(
        features.get('principal_direction_extents', m, direction=0),
        [
            1210.569727,
            117.988454,
            147.098687,
            288.226628,
            330.166506,
            152.396521,
            293.913857,
        ],
        atol=1e-6
    )
    assert_allclose(
        features.get('principal_direction_extents', m, direction=1),
        [
            851.730088,
            99.108911,
            116.949436,
            157.171734,
            137.328019,
            20.66982,
            67.157249,
        ],
        atol=1e-6
    )
    assert_allclose(
        features.get('principal_direction_extents', m, direction=2),
        [
            282.961199,
            38.493958,
            40.715183,
            94.061625,
            51.120255,
            10.793167,
            62.808188
        ],
        atol=1e-6
    )

def test_total_width():

    assert_allclose(
        features.get('total_width', NRN),
        105.0758
    )

    assert_allclose(
        features.get('total_width', NRN, neurite_type=nm.AXON),
        33.25306
    )

    assert_allclose(
        features.get('total_width', NRN, neurite_type=nm.BASAL_DENDRITE),
        104.57807
    )


def test_total_height():

    assert_allclose(
        features.get('total_height', NRN),
        106.11643
    )

    assert_allclose(
        features.get('total_height', NRN, neurite_type=nm.AXON),
        57.60017
    )

    assert_allclose(
        features.get('total_height', NRN, neurite_type=nm.BASAL_DENDRITE),
        48.516262
    )

def test_total_depth():

    assert_allclose(
        features.get('total_depth', NRN),
        54.204086
    )

    assert_allclose(
        features.get('total_depth', NRN, neurite_type=nm.AXON),
        49.70138
    )

    assert_allclose(
        features.get('total_depth', NRN, neurite_type=nm.BASAL_DENDRITE),
        51.64143
    )


def test_aspect_ratio():

    morph = load_morphology(DATA_PATH / "neurolucida/bio_neuron-000.asc")

    npt.assert_almost_equal(
        features.get("aspect_ratio", morph, neurite_type=nm.AXON, projection_plane="xy"),
        0.710877,
        decimal=6
    )
    npt.assert_almost_equal(
        features.get("aspect_ratio", morph, neurite_type=nm.AXON, projection_plane="xz"),
        0.222268,
        decimal=6
    )
    npt.assert_almost_equal(
        features.get("aspect_ratio", morph, neurite_type=nm.AXON, projection_plane="yz"),
        0.315263,
        decimal=6
    )
    npt.assert_almost_equal(
        features.get("aspect_ratio", morph),
        0.731076,
        decimal=6
    )
    assert np.isnan(features.get("aspect_ratio", morph, neurite_type=nm.NeuriteType.custom5))


def test_circularity():

    morph = load_morphology(DATA_PATH / "neurolucida/bio_neuron-000.asc")

    npt.assert_almost_equal(
        features.get("circularity", morph, neurite_type=nm.AXON, projection_plane="xy"),
        0.722613,
        decimal=6
    )
    npt.assert_almost_equal(
        features.get("circularity", morph, neurite_type=nm.AXON, projection_plane="xz"),
        0.378692,
        decimal=6
    )
    npt.assert_almost_equal(
        features.get("circularity", morph, neurite_type=nm.AXON, projection_plane="yz"),
        0.527657,
        decimal=6
    )
    npt.assert_almost_equal(
        features.get("circularity", morph),
        0.730983,
        decimal=6
    )
    assert np.isnan(features.get("circularity", morph, neurite_type=nm.NeuriteType.custom5))


def test_shape_factor():

    morph = load_morphology(DATA_PATH / "neurolucida/bio_neuron-000.asc")

    npt.assert_almost_equal(
        features.get("shape_factor", morph, neurite_type=nm.AXON, projection_plane="xy"),
        0.356192,
        decimal=6
    )
    npt.assert_almost_equal(
        features.get("shape_factor", morph, neurite_type=nm.AXON, projection_plane="xz"),
        0.131547,
        decimal=6
    )
    npt.assert_almost_equal(
        features.get("shape_factor", morph, neurite_type=nm.AXON, projection_plane="yz"),
        0.194558,
        decimal=6
    )
    npt.assert_almost_equal(
        features.get("shape_factor", morph),
        0.364678,
        decimal=6
    )
    assert np.isnan(features.get("shape_factor", morph, neurite_type=nm.NeuriteType.custom5))


@pytest.mark.parametrize("neurite_type, axis, expected_value", [
    (nm.AXON, "X", 0.50),
    (nm.AXON, "Y", 0.74),
    (nm.AXON, "Z", 0.16),
    (nm.APICAL_DENDRITE, "X", np.nan),
    (nm.APICAL_DENDRITE, "Y", np.nan),
    (nm.APICAL_DENDRITE, "Z", np.nan),
    (nm.BASAL_DENDRITE, "X", 0.50),
    (nm.BASAL_DENDRITE, "Y", 0.59),
    (nm.BASAL_DENDRITE, "Z", 0.48),
]
)
def test_length_fraction_from_soma(neurite_type, axis, expected_value):

    morph = load_morphology(DATA_PATH / "neurolucida/bio_neuron-000.asc")

    npt.assert_almost_equal(
        features.get("length_fraction_above_soma", morph, neurite_type=neurite_type, up=axis),
        expected_value,
        decimal=2
    )


def test_length_fraction_from_soma__wrong_axis():

    morph = load_morphology(DATA_PATH / "neurolucida/bio_neuron-000.asc")

    with pytest.raises(NeuroMError):
        features.get("length_fraction_above_soma", morph, up='K')
