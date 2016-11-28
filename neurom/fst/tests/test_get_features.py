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

'''Test neurom.fst_get features'''

import os
import math
import numpy as np
from nose import tools as nt
from neurom.core.types import NeuriteType
from neurom.core.population import Population
from neurom import core, load_neurons, iter_neurites
from neurom import fst
from neurom.fst import get as fst_get
from neurom.fst import NEURITEFEATURES
from neurom.core.types import tree_type_checker as _is_type
from neurom.exceptions import NeuroMError


_PWD = os.path.dirname(os.path.abspath(__file__))
NRN_FILES = [os.path.join(_PWD, '../../../test_data/h5/v1', f)
             for f in ('Neuron.h5', 'Neuron_2_branch.h5', 'bio_neuron-001.h5')]

NRNS = load_neurons(NRN_FILES)
NRN = NRNS[0]
POP = Population(NRNS)

NEURITES = (NeuriteType.axon,
            NeuriteType.apical_dendrite,
            NeuriteType.basal_dendrite,
            NeuriteType.all)


def assert_items_equal(a, b):
    nt.eq_(sorted(a), sorted(b))


def _stats(seq):
    return np.min(seq), np.max(seq), np.sum(seq), np.mean(seq)

def test_number_of_sections_pop():

    feat = 'number_of_sections'

    assert_items_equal(fst_get(feat, POP),
                          [84, 42, 202])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.all),
                          [84, 42, 202])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.axon),
                          [21, 21, 179])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                          [21, 0, 0])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                          [42, 21, 23])


def test_number_of_sections_nrn():

    feat = 'number_of_sections'

    assert_items_equal(fst_get(feat, NRN),
                          [84])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.all),
                          [84])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.axon),
                          [21])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                          [21])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                          [42])


def test_section_tortuosity_pop():

    feat = 'section_tortuosity'

    nt.ok_(np.allclose(_stats(fst_get(feat, POP)),
                       (1.0,
                        4.6571118550276704,
                        440.40884450374455,
                        1.3427098917797089)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.all)),
                       (1.0,
                        4.6571118550276704,
                        440.40884450374455,
                        1.3427098917797089)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                       (1.0702760052031615,
                        1.5732825321954913,
                        26.919574286670883,
                        1.2818844898414707)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                       (1.042614577410971,
                        1.6742599832295344,
                        106.5960839640893,
                        1.239489348419643)))


def test_section_tortuosity_nrn():

    feat = 'section_tortuosity'


def test_number_of_segments_pop():

    feat = 'number_of_segments'

    assert_items_equal(fst_get(feat, POP), [840, 419, 5179])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.all),
                          [840, 419, 5179])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.axon),
                          [210, 209, 4508])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                          [210, 0, 0])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                          [420, 210, 671])


def test_number_of_segments_nrn():

    feat = 'number_of_segments'

    assert_items_equal(fst_get(feat, NRN), [840])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.all),
                          [840])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.axon),
                          [210])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                          [210])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                          [420])


def test_number_of_neurites_pop():

    feat = 'number_of_neurites'

    assert_items_equal(fst_get(feat, POP), [4, 2, 4])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.all),
                          [4, 2, 4])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.axon),
                          [1, 1, 1])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                          [1, 0, 0])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                          [2, 1, 3])


def test_number_of_neurites_nrn():

    feat = 'number_of_neurites'

    assert_items_equal(fst_get(feat, NRN), [4])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.all),
                          [4])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.axon),
                          [1])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                          [1])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                          [2])


def test_number_of_bifurcations_nrn():

    feat = 'number_of_bifurcations'

    assert_items_equal(fst_get(feat, NRN), [40])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.all),
                          [40])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.axon),
                          [10])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                          [10])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                          [20])


def test_number_of_bifurcations_pop():

    feat = 'number_of_bifurcations'

    assert_items_equal(fst_get(feat, POP), [40, 20, 97])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.all),
                          [40, 20, 97])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.axon),
                          [10, 10, 87])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                          [10, 0, 0])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                          [20, 10, 10])


def test_number_of_forking_points_nrn():

    feat = 'number_of_forking_points'

    assert_items_equal(fst_get(feat, NRN), [40])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.all),
                          [40])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.axon),
                          [10])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                          [10])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                          [20])


def test_number_of_forking_points_pop():

    feat = 'number_of_forking_points'

    assert_items_equal(fst_get(feat, POP), [40, 20, 98])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.all),
                          [40, 20, 98])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.axon),
                          [10, 10, 88])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                          [10, 0, 0])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                          [20, 10, 10])


def test_number_of_terminations_nrn():

    feat = 'number_of_terminations'

    assert_items_equal(fst_get(feat, NRN), [44])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.all),
                          [44])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.axon),
                          [11])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                          [11])

    assert_items_equal(fst_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                          [22])


def test_number_of_terminations_pop():

    feat = 'number_of_terminations'

    assert_items_equal(fst_get(feat, POP), [44, 22, 103])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.all),
                          [44, 22, 103])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.axon),
                          [11, 11, 90])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                          [11, 0, 0])

    assert_items_equal(fst_get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                          [22, 11, 13])


def test_total_length_pop():

    feat = 'total_length'

    nt.ok_(np.allclose(fst_get(feat, POP),
                       [840.68522362, 418.83424432, 13250.82577394]))

    nt.ok_(np.allclose(fst_get(feat, POP, neurite_type=NeuriteType.all),
                       [840.68522362, 418.83424432, 13250.82577394]))

    nt.ok_(np.allclose(fst_get(feat, POP, neurite_type=NeuriteType.axon),
                       [207.8797736, 207.81088342, 11767.15611522]))

    nt.ok_(np.allclose(fst_get(feat, POP, neurite_type=NeuriteType.apical_dendrite),
                       [214.37302709, 0, 0]))

    nt.ok_(np.allclose(fst_get(feat, POP, neurite_type=NeuriteType.basal_dendrite),
                       [418.43242293, 211.0233609, 1483.66965872]))


def test_total_length_nrn():

    feat = 'total_length'

    nt.ok_(np.allclose(fst_get(feat, NRN),
                       [840.68522362]))

    nt.ok_(np.allclose(fst_get(feat, NRN, neurite_type=NeuriteType.all),
                       [840.68522362]))

    nt.ok_(np.allclose(fst_get(feat, NRN, neurite_type=NeuriteType.axon),
                       [207.8797736]))

    nt.ok_(np.allclose(fst_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite),
                       [214.37302709]))

    nt.ok_(np.allclose(fst_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite),
                       [418.43242293]))


def test_segment_radii_pop():

    feat = 'segment_radii'

    nt.ok_(np.allclose(_stats(fst_get(feat, POP)),
                       (0.079999998211860657,
                        1.2150000333786011,
                        1301.9191725363567,
                        0.20222416473071708)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.all)),
                       (0.079999998211860657,
                        1.2150000333786011,
                        1301.9191725363567,
                        0.20222416473071708)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                       (0.13142434507608414,
                        1.0343990325927734,
                        123.41135908663273,
                        0.58767313850777492)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                       (0.079999998211860657,
                        1.2150000333786011,
                        547.43900821779164,
                        0.42078324997524336)))


def test_segment_radii_nrn():

    feat = 'segment_radii'

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN)),
                       (0.12087134271860123,
                        1.0343990325927734,
                        507.01994501426816,
                        0.60359517263603357)))

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.all)),
                       (0.12087134271860123,
                        1.0343990325927734,
                        507.01994501426816,
                        0.60359517263603357)))

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                       (0.13142434507608414,
                        1.0343990325927734,
                        123.41135908663273,
                        0.58767313850777492)))

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                       (0.14712842553853989,
                        1.0215770602226257,
                        256.71241207793355,
                        0.61122002875698467)))


def test_segment_meander_angles_pop():

    feat = 'segment_meander_angles'

    nt.ok_(np.allclose(_stats(fst_get(feat, POP)),
                       (0.0, 3.1415926535897931, 14637.977670710961, 2.3957410263029395)
                       ))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.all)),
                       (0.0, 3.1415926535897931, 14637.977670710961, 2.3957410263029395)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                       (0.326101999292573, 3.0939261437163492, 461.98168732359414, 2.4443475519766884)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                       (0.0, 3.1415926535897931, 2926.2411975307768, 2.4084289691611334)))


def test_segment_meander_angles_nrn():

    feat = 'segment_meander_angles'

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN)),
                       (0.326101999292573, 3.129961675751181, 1842.351779156608, 2.4369732528526562)))

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.all)),
                       (0.326101999292573, 3.129961675751181, 1842.351779156608, 2.4369732528526562)))

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                       (0.326101999292573, 3.0939261437163492, 461.98168732359414, 2.4443475519766884)))

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                       (0.47318725279312024, 3.129961675751181, 926.33847274926438, 2.4506308802890593)))


def test_neurite_volumes_nrn():

    feat = 'neurite_volumes'

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN)),
                       (271.94122143951864, 281.24754646913954, 1104.9077698137021, 276.22694245342552)))

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.all)),
                       (271.94122143951864, 281.24754646913954, 1104.9077698137021, 276.22694245342552)))

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.axon)),
                       (276.73860261723024, 276.73860261723024, 276.73860261723024, 276.73860261723024)))


    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                       (274.98039928781355, 281.24754646913954, 556.22794575695309, 278.11397287847655)))


    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                       (271.94122143951864, 271.94122143951864, 271.94122143951864, 271.94122143951864)))


def test_neurite_volumes_pop():

    feat = 'neurite_volumes'


    nt.ok_(np.allclose(_stats(fst_get(feat, POP)),
                       (28.356406629821159, 281.24754646913954, 2249.4613918388391, 224.9461391838839)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.all)),
                       (28.356406629821159, 281.24754646913954, 2249.4613918388391, 224.9461391838839)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.axon)),
                       (276.58135508666612, 277.5357232437392, 830.85568094763551, 276.95189364921185)))


    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                       (28.356406629821159, 281.24754646913954, 1146.6644894516851, 191.1107482419475)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                       (271.94122143951864, 271.94122143951864, 271.94122143951864, 271.94122143951864)))


def test_neurite_density_nrn():

    feat = 'neurite_volume_density'

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN)),
                       (0.24068543213643726, 0.52464681266899216, 1.4657913638494682, 0.36644784096236704)))

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.all)),
                       (0.24068543213643726, 0.52464681266899216, 1.4657913638494682, 0.36644784096236704)))

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.axon)),
                       (0.26289304906104355, 0.26289304906104355, 0.26289304906104355, 0.26289304906104355)))

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                       (0.24068543213643726, 0.52464681266899216, 0.76533224480542938, 0.38266612240271469)))

    nt.ok_(np.allclose(_stats(fst_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                       (0.43756606998299519, 0.43756606998299519, 0.43756606998299519, 0.43756606998299519)))


def test_neurite_density_pop():

    feat = 'neurite_volume_density'

    nt.ok_(np.allclose(_stats(fst_get(feat, POP)),
                       (6.1847539631150784e-06, 0.52464681266899216, 1.9767794901940539, 0.19767794901940539)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.all)),
                       (6.1847539631150784e-06, 0.52464681266899216, 1.9767794901940539, 0.19767794901940539)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.axon)),
                       (6.1847539631150784e-06, 0.26465213325053372, 0.5275513670655404, 0.17585045568851346)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                       (0.00034968816544949771, 0.52464681266899216, 1.0116620531455183, 0.16861034219091972)))

    nt.ok_(np.allclose(_stats(fst_get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                       (0.43756606998299519, 0.43756606998299519, 0.43756606998299519, 0.43756606998299519)))


def test_segment_meander_angles_single_section():

    class Mock(object):
        pass

    feat = 'segment_meander_angles'

    sec = core.Section(np.array([[0, 0, 0],
                                 [1, 0, 0],
                                 [1, 1, 0],
                                 [2, 1, 0],
                                 [2, 2, 0]]))

    nrt = core.Neurite(sec)
    nrn = Mock()
    nrn.neurites = [nrt]
    nrn.soma = None
    pop = core.Population([nrn])

    ref = [math.pi / 2, math.pi / 2, math.pi / 2]

    nt.assert_equal(ref, fst_get(feat, nrt).tolist())
    nt.assert_equal(ref, fst_get(feat, nrn).tolist())
    nt.assert_equal(ref, fst_get(feat, pop).tolist())


def test_neurite_features_accept_single_tree():

    features = NEURITEFEATURES.keys()

    for f in features:
        fst_get(f, NRN.neurites[0])


def test_register_neurite_feature_nrns():

    def npts(neurite):
        return len(neurite.points)

    def vol(neurite):
        return neurite.volume

    fst.register_neurite_feature('foo', npts)

    n_points_ref = [len(n.points) for n in iter_neurites(NRNS)]
    n_points = fst.get('foo', NRNS)
    assert_items_equal(n_points, n_points_ref)

    # test neurite type filtering
    n_points_ref = [len(n.points) for n in iter_neurites(NRNS, filt=_is_type(NeuriteType.axon))]
    n_points = fst.get('foo', NRNS, neurite_type=NeuriteType.axon)
    assert_items_equal(n_points, n_points_ref)

    fst.register_neurite_feature('bar', vol)

    n_volume_ref = [n.volume for n in iter_neurites(NRNS)]
    n_volume = fst.get('bar', NRNS)
    assert_items_equal(n_volume, n_volume_ref)

    # test neurite type filtering
    n_volume_ref = [n.volume for n in iter_neurites(NRNS, filt=_is_type(NeuriteType.axon))]
    n_volume = fst.get('bar', NRNS, neurite_type=NeuriteType.axon)
    assert_items_equal(n_volume, n_volume_ref)


def test_register_neurite_feature_pop():

    def npts(neurite):
        return len(neurite.points)

    def vol(neurite):
        return neurite.volume

    fst.register_neurite_feature('foo', npts)

    n_points_ref = [len(n.points) for n in iter_neurites(POP)]
    n_points = fst.get('foo', POP)
    assert_items_equal(n_points, n_points_ref)

    # test neurite type filtering
    n_points_ref = [len(n.points) for n in iter_neurites(POP,
                                                         filt=_is_type(NeuriteType.basal_dendrite))]
    n_points = fst.get('foo', POP, neurite_type=NeuriteType.basal_dendrite)
    assert_items_equal(n_points, n_points_ref)

    fst.register_neurite_feature('bar', vol)

    n_volume_ref = [n.volume for n in iter_neurites(POP)]
    n_volume = fst.get('bar', POP)
    assert_items_equal(n_volume, n_volume_ref)

    # test neurite type filtering
    n_volume_ref = [n.volume for n in iter_neurites(POP, filt=_is_type(NeuriteType.basal_dendrite))]
    n_volume = fst.get('bar', POP, neurite_type=NeuriteType.basal_dendrite)
    assert_items_equal(n_volume, n_volume_ref)


@nt.raises(NeuroMError)
def test_register_existing_feature_raises():
    fst.register_neurite_feature('total_length', lambda n: None)
