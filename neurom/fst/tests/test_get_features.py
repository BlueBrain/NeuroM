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

'''Test neurom.fst.get features'''

import os
import math
import numpy as np
from nose import tools as nt
from neurom.core.types import NeuriteType
from neurom.core.population import Population
from neurom import fst


_PWD = os.path.dirname(os.path.abspath(__file__))
NRN_FILES = [os.path.join(_PWD, '../../../test_data/h5/v1', f)
             for f in ('Neuron.h5', 'Neuron_2_branch.h5', 'bio_neuron-001.h5')]

NRNS = fst.load_neurons(NRN_FILES)
NRN = NRNS[0]
POP = Population(NRNS)

NEURITES = (NeuriteType.axon,
            NeuriteType.apical_dendrite,
            NeuriteType.basal_dendrite,
            NeuriteType.all)


def _stats(seq):
    return np.min(seq), np.max(seq), np.sum(seq), np.mean(seq)

def test_number_of_sections_pop():

    feat = 'number_of_sections'

    nt.assert_items_equal(fst.get(feat, POP), [84, 42, 202])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.all),
                          [84, 42, 202])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.axon),
                          [21, 21, 179])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                          [21, 0, 0])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                          [42, 21, 23])


def test_number_of_sections_nrn():

    feat = 'number_of_sections'

    nt.assert_items_equal(fst.get(feat, NRN), [84])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.all),
                          [84])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.axon),
                          [21])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                          [21])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                          [42])


def test_section_tortuosity_pop():

    feat = 'section_tortuosity'

    nt.ok_(np.allclose(_stats(fst.get(feat, POP)),
                       (1.0,
                        4.6571118550276704,
                        440.40884450374455,
                        1.3427098917797089)))

    nt.ok_(np.allclose(_stats(fst.get(feat, POP, neurite_type=NeuriteType.all)),
                       (1.0,
                        4.6571118550276704,
                        440.40884450374455,
                        1.3427098917797089)))

    nt.ok_(np.allclose(_stats(fst.get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                       (1.0702760052031615,
                        1.5732825321954913,
                        26.919574286670883,
                        1.2818844898414707)))

    nt.ok_(np.allclose(_stats(fst.get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                       (1.042614577410971,
                        1.6742599832295344,
                        106.5960839640893,
                        1.239489348419643)))


def test_section_tortuosity_nrn():

    feat = 'section_tortuosity'


def test_number_of_segments_pop():

    feat = 'number_of_segments'

    nt.assert_items_equal(fst.get(feat, POP), [840, 419, 5179])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.all),
                          [840, 419, 5179])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.axon),
                          [210, 209, 4508])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                          [210, 0, 0])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                          [420, 210, 671])


def test_number_of_segments_nrn():

    feat = 'number_of_segments'

    nt.assert_items_equal(fst.get(feat, NRN), [840])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.all),
                          [840])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.axon),
                          [210])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                          [210])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                          [420])


def test_number_of_neurites_pop():

    feat = 'number_of_neurites'

    nt.assert_items_equal(fst.get(feat, POP), [4, 2, 4])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.all),
                          [4, 2, 4])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.axon),
                          [1, 1, 1])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                          [1, 0, 0])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                          [2, 1, 3])


def test_number_of_neurites_nrn():

    feat = 'number_of_neurites'

    nt.assert_items_equal(fst.get(feat, NRN), [4])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.all),
                          [4])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.axon),
                          [1])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                          [1])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                          [2])


def test_number_of_bifurcations_nrn():

    feat = 'number_of_bifurcations'

    nt.assert_items_equal(fst.get(feat, NRN), [40])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.all),
                          [40])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.axon),
                          [10])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                          [10])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                          [20])


def test_number_of_bifurcations_pop():

    feat = 'number_of_bifurcations'

    nt.assert_items_equal(fst.get(feat, POP), [40, 20, 97])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.all),
                          [40, 20, 97])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.axon),
                          [10, 10, 87])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                          [10, 0, 0])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                          [20, 10, 10])


def test_number_of_forking_points_nrn():

    feat = 'number_of_forking_points'

    nt.assert_items_equal(fst.get(feat, NRN), [40])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.all),
                          [40])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.axon),
                          [10])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                          [10])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                          [20])


def test_number_of_forking_points_pop():

    feat = 'number_of_forking_points'

    nt.assert_items_equal(fst.get(feat, POP), [40, 20, 98])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.all),
                          [40, 20, 98])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.axon),
                          [10, 10, 88])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                          [10, 0, 0])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                          [20, 10, 10])


def test_number_of_terminations_nrn():

    feat = 'number_of_terminations'

    nt.assert_items_equal(fst.get(feat, NRN), [44])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.all),
                          [44])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.axon),
                          [11])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                          [11])

    nt.assert_items_equal(fst.get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                          [22])


def test_number_of_terminations_pop():

    feat = 'number_of_terminations'

    nt.assert_items_equal(fst.get(feat, POP), [44, 22, 103])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.all),
                          [44, 22, 103])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.axon),
                          [11, 11, 90])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                          [11, 0, 0])

    nt.assert_items_equal(fst.get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                          [22, 11, 13])


def test_total_length_pop():

    feat = 'total_length'

    nt.ok_(np.allclose(fst.get(feat, POP),
                       [840.68522362, 418.83424432, 13250.82577394]))

    nt.ok_(np.allclose(fst.get(feat, POP, neurite_type=NeuriteType.all),
                       [840.68522362, 418.83424432, 13250.82577394]))

    nt.ok_(np.allclose(fst.get(feat, POP, neurite_type=NeuriteType.axon),
                       [207.8797736, 207.81088342, 11767.15611522]))

    nt.ok_(np.allclose(fst.get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                       [214.37302709, 0, 0]))

    nt.ok_(np.allclose(fst.get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                       [418.43242293, 211.0233609, 1483.66965872]))


def test_total_length_nrn():

    feat = 'total_length'

    nt.ok_(np.allclose(fst.get(feat, NRN),
                       [840.68522362]))

    nt.ok_(np.allclose(fst.get(feat, NRN, neurite_type=NeuriteType.all),
                       [840.68522362]))

    nt.ok_(np.allclose(fst.get(feat, NRN, neurite_type=NeuriteType.axon),
                       [207.8797736]))

    nt.ok_(np.allclose(fst.get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                       [214.37302709]))

    nt.ok_(np.allclose(fst.get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                       [418.43242293]))


def test_segment_radii_pop():

    feat = 'segment_radii'

    nt.ok_(np.allclose(_stats(fst.get(feat, POP)),
                       (0.079999998211860657,
                        1.2150000333786011,
                        1301.9191725363567,
                        0.20222416473071708)))

    nt.ok_(np.allclose(_stats(fst.get(feat, POP, neurite_type=NeuriteType.all)),
                       (0.079999998211860657,
                        1.2150000333786011,
                        1301.9191725363567,
                        0.20222416473071708)))

    nt.ok_(np.allclose(_stats(fst.get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                       (0.13142434507608414,
                        1.0343990325927734,
                        123.41135908663273,
                        0.58767313850777492)))

    nt.ok_(np.allclose(_stats(fst.get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                       (0.079999998211860657,
                        1.2150000333786011,
                        547.43900821779164,
                        0.42078324997524336)))


def test_segment_radii_nrn():

    feat = 'segment_radii'

    nt.ok_(np.allclose(_stats(fst.get(feat, NRN)),
                       (0.12087134271860123,
                        1.0343990325927734,
                        507.01994501426816,
                        0.60359517263603357)))

    nt.ok_(np.allclose(_stats(fst.get(feat, NRN, neurite_type=NeuriteType.all)),
                       (0.12087134271860123,
                        1.0343990325927734,
                        507.01994501426816,
                        0.60359517263603357)))

    nt.ok_(np.allclose(_stats(fst.get(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                       (0.13142434507608414,
                        1.0343990325927734,
                        123.41135908663273,
                        0.58767313850777492)))

    nt.ok_(np.allclose(_stats(fst.get(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                       (0.14712842553853989,
                        1.0215770602226257,
                        256.71241207793355,
                        0.61122002875698467)))


def test_segment_meander_angles_pop():

    feat = 'segment_meander_angles'

    nt.ok_(np.allclose(_stats(fst.get(feat, POP)),
                       (0.0, 3.1415926535897931, 14637.977670710961, 2.3957410263029395)
                       ))

    nt.ok_(np.allclose(_stats(fst.get(feat, POP, neurite_type=NeuriteType.all)),
                       (0.0, 3.1415926535897931, 14637.977670710961, 2.3957410263029395)))

    nt.ok_(np.allclose(_stats(fst.get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                       (0.326101999292573, 3.0939261437163492, 461.98168732359414, 2.4443475519766884)))

    nt.ok_(np.allclose(_stats(fst.get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                       (0.0, 3.1415926535897931, 2926.2411975307768, 2.4084289691611334)))


def test_segment_meander_angles_nrn():

    feat = 'segment_meander_angles'

    nt.ok_(np.allclose(_stats(fst.get(feat, NRN)),
                       (0.326101999292573, 3.129961675751181, 1842.351779156608, 2.4369732528526562)))

    nt.ok_(np.allclose(_stats(fst.get(feat, NRN, neurite_type=NeuriteType.all)),
                       (0.326101999292573, 3.129961675751181, 1842.351779156608, 2.4369732528526562)))

    nt.ok_(np.allclose(_stats(fst.get(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                       (0.326101999292573, 3.0939261437163492, 461.98168732359414, 2.4443475519766884)))

    nt.ok_(np.allclose(_stats(fst.get(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                       (0.47318725279312024, 3.129961675751181, 926.33847274926438, 2.4506308802890593)))


def test_segment_meander_angles_single_section():

    class Mock(object):
        pass

    feat = 'segment_meander_angles'

    sec = fst.Section(np.array([[0, 0, 0],
                                [1, 0, 0],
                                [1, 1, 0],
                                [2, 1, 0],
                                [2, 2, 0]]))

    nrt = fst.Neurite(sec)
    nrn = Mock()
    nrn.neurites = [nrt]
    nrn.soma = None
    pop = fst.Population([nrn])

    ref = [math.pi / 2, math.pi / 2, math.pi / 2]

    nt.assert_equal(ref, fst.get(feat, nrt).tolist())
    nt.assert_equal(ref, fst.get(feat, nrn).tolist())
    nt.assert_equal(ref, fst.get(feat, pop).tolist())


def test_neurite_features_accept_single_tree():

    features = fst.NEURITEFEATURES.keys()

    for f in features:
        fst.get(f, NRN.neurites[0])
