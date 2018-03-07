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

'''Test neurom.get features'''

import os
import math
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from nose import tools as nt
import neurom as nm
from neurom.core.types import NeuriteType
from neurom.core.population import Population
from neurom import (core, load_neurons, iter_neurites, iter_sections,
                    load_neuron, features)
from neurom.features import get as features_get, FEATURES, neuritefunc as nf, _get_doc
from neurom.core.types import tree_type_checker as _is_type
from neurom.exceptions import NeuroMError


_PWD = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_PWD, '../../../test_data')
NRN_FILES = [os.path.join(DATA_PATH, 'h5/v1', f)
             for f in ('Neuron.h5', 'Neuron_2_branch.h5', 'bio_neuron-001.h5')]

NRNS = load_neurons(NRN_FILES)
NRN = NRNS[0]
POP = Population(NRNS)

NEURITES = (NeuriteType.axon,
            NeuriteType.apical_dendrite,
            NeuriteType.basal_dendrite)


def assert_items_equal(a, b):
    nt.eq_(sorted(a), sorted(b))


def assert_features_for_neurite(feat, neurons, expected, exact=True):
    for neurite_type, expected_values in expected.items():
        if neurite_type is None:
            res_pop = features_get(feat, neurons)
            res = features_get(feat, neurons[0])
        else:
            res_pop = features_get(feat, neurons, neurite_type=neurite_type)
            res = features_get(feat, neurons[0], neurite_type=neurite_type)

        if exact:
            assert_items_equal(res_pop, expected_values)
        else:
            assert_allclose(res_pop, expected_values, rtol=1e-5)

        #test for single neuron
        if isinstance(res, np.ndarray):
            # some features, ex: total_length return arrays w/ one element when
            # called on a single neuron
            nt.eq_(len(res), 1)
            res = res[0]
        if exact:
            nt.eq_(res, expected_values[0])
        else:
            assert_allclose(res, expected_values[0])


def _stats(seq):
    return np.min(seq), np.max(seq), np.sum(seq), np.mean(seq)


def test_number_of_sections():
    feat = 'number_of_sections'
    expected = {None: [84, 42, 202],
                NeuriteType.axon: [21, 21, 179],
                NeuriteType.apical_dendrite: [21, 0, 0],
                NeuriteType.basal_dendrite: [42, 21, 23],
                }
    assert_features_for_neurite(feat, POP, expected)


def test_section_tortuosity_pop():

    feat = 'section_tortuosity'

    assert_allclose(_stats(features_get(feat, POP)),
                    (1.0,
                     4.6571118550276704,
                     440.40884450374455,
                     1.3427098917797089))

    assert_allclose(_stats(features_get(feat, POP, neurite_type=None)),
                    (1.0,
                     4.6571118550276704,
                     440.40884450374455,
                     1.3427098917797089))

    assert_allclose(_stats(features_get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                    (1.0702760052031615,
                     1.5732825321954913,
                     26.919574286670883,
                     1.2818844898414707))

    assert_allclose(_stats(features_get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                    (1.042614577410971,
                     1.6742599832295344,
                     106.5960839640893,
                     1.239489348419643),
                    rtol=1e-5)


def test_section_tortuosity_nrn():
    feat = 'section_tortuosity'
    nt.ok_(np.allclose(_stats(features_get(feat, NRN)),
                       (1.0702760052031612,
                        1.5732825321954911,
                        106.42449427885093,
                        1.2669582652244158)))

    nt.ok_(np.allclose(_stats(features_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                       (1.0702760052031612,
                        1.5732825321954911,
                        26.919574286670883,
                        1.2818844898414707)))

    nt.ok_(np.allclose(_stats(features_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                       (1.0788578286296124,
                        1.5504287518256337,
                        51.540901640170489,
                        1.227164324765964)))


def test_number_of_segments():

    feat = 'number_of_segments'

    expected = {None: [840, 419, 5179],
                NeuriteType.axon: [210, 209, 4508],
                NeuriteType.apical_dendrite: [210, 0, 0],
                NeuriteType.basal_dendrite: [420, 210, 671],
                }

    assert_features_for_neurite(feat, POP, expected)


def test_number_of_neurites_pop():
    feat = 'number_of_neurites'
    expected = {None: [4, 2, 4],
                NeuriteType.axon: [1, 1, 1],
                NeuriteType.apical_dendrite: [1, 0, 0],
                NeuriteType.basal_dendrite: [2, 1, 3],
                }
    assert_features_for_neurite(feat, POP, expected)


def test_number_of_bifurcations_pop():
    feat = 'number_of_bifurcations'
    expected = {None: [40, 20, 97],
                NeuriteType.axon: [10, 10, 87],
                NeuriteType.apical_dendrite: [10, 0, 0],
                NeuriteType.basal_dendrite: [20, 10, 10],
                }
    assert_features_for_neurite(feat, POP, expected)


def test_number_of_forking_points_pop():

    feat = 'number_of_forking_points'

    expected = {None: [40, 20, 98],
                NeuriteType.axon: [10, 10, 88],
                NeuriteType.apical_dendrite: [10, 0, 0],
                NeuriteType.basal_dendrite: [20, 10, 10],
                }
    assert_features_for_neurite(feat, POP, expected)


def test_number_of_terminations_pop():
    feat = 'number_of_terminations'
    expected = {None: [44, 22, 103],
                NeuriteType.axon: [11, 11, 90],
                NeuriteType.apical_dendrite: [11, 0, 0],
                NeuriteType.basal_dendrite: [22, 11, 13],
                }
    assert_features_for_neurite(feat, POP, expected)


def test_total_length_pop():
    feat = 'total_length'
    expected = {None: [840.68522362011538, 418.83424432013902, 13250.825773939932],
                NeuriteType.axon: [207.8797736031714, 207.81088341560977, 11767.156115224638],
                NeuriteType.apical_dendrite: [214.37302709169489, 0, 0],
                NeuriteType.basal_dendrite: [418.43242292524889, 211.02336090452931, 1483.6696587152967],
                }
    assert_features_for_neurite(feat, POP, expected, exact=False)

def test_segment_radii_pop():

    feat = 'segment_radii'

    nt.ok_(np.allclose(_stats(features_get(feat, POP)),
                       (0.079999998211860657,
                        1.2150000333786011,
                        1301.9191725363567,
                        0.20222416473071708)))

    nt.ok_(np.allclose(_stats(features_get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                       (0.13142434507608414,
                        1.0343990325927734,
                        123.41135908663273,
                        0.58767313850777492)))

    nt.ok_(np.allclose(_stats(features_get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                       (0.079999998211860657,
                        1.2150000333786011,
                        547.43900821779164,
                        0.42078324997524336)))


def test_segment_radii_nrn():

    feat = 'segment_radii'

    nt.ok_(np.allclose(_stats(features_get(feat, NRN)),
                       (0.12087134271860123,
                        1.0343990325927734,
                        507.01994501426816,
                        0.60359517263603357)))

    nt.ok_(np.allclose(_stats(features_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                       (0.13142434507608414,
                        1.0343990325927734,
                        123.41135908663273,
                        0.58767313850777492)))

    nt.ok_(np.allclose(_stats(features_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                       (0.14712842553853989,
                        1.0215770602226257,
                        256.71241207793355,
                        0.61122002875698467)))


def test_segment_meander_angles_pop():

    feat = 'segment_meander_angles'

    nt.ok_(np.allclose(_stats(features_get(feat, POP)),
                       (0.0, 3.1415926535897931, 14637.977670710961, 2.3957410263029395)))

    nt.ok_(np.allclose(_stats(features_get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                       (0.326101999292573, 3.0939261437163492, 461.98168732359414, 2.4443475519766884)))

    nt.ok_(np.allclose(_stats(features_get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                       (0.0, 3.1415926535897931, 2926.2411975307768, 2.4084289691611334)))


def test_segment_meander_angles_nrn():

    feat = 'segment_meander_angles'

    nt.ok_(np.allclose(_stats(features_get(feat, NRN)),
                       (0.326101999292573, 3.129961675751181, 1842.351779156608, 2.4369732528526562)))

    nt.ok_(np.allclose(_stats(features_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                       (0.326101999292573, 3.0939261437163492, 461.98168732359414, 2.4443475519766884)))

    nt.ok_(np.allclose(_stats(features_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                       (0.47318725279312024, 3.129961675751181, 926.33847274926438, 2.4506308802890593)))


def test_neurite_volumes_nrn():

    feat = 'neurite_volumes'

    nt.ok_(np.allclose(_stats(features_get(feat, NRN)),
                       (271.94122143951864, 281.24754646913954, 1104.9077698137021, 276.22694245342552)))

    nt.ok_(np.allclose(_stats(features_get(feat, NRN, neurite_type=NeuriteType.axon)),
                       (276.73860261723024, 276.73860261723024, 276.73860261723024, 276.73860261723024)))


    nt.ok_(np.allclose(_stats(features_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                       (274.98039928781355, 281.24754646913954, 556.22794575695309, 278.11397287847655)))


    nt.ok_(np.allclose(_stats(features_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                       (271.94122143951864, 271.94122143951864, 271.94122143951864, 271.94122143951864)))


def test_neurite_volumes_pop():

    feat = 'neurite_volumes'


    nt.ok_(np.allclose(_stats(features_get(feat, POP)),
                       (28.356406629821159, 281.24754646913954, 2249.4613918388391, 224.9461391838839)))

    nt.ok_(np.allclose(_stats(features_get(feat, POP, neurite_type=NeuriteType.axon)),
                       (276.58135508666612, 277.5357232437392, 830.85568094763551, 276.95189364921185)))


    nt.ok_(np.allclose(_stats(features_get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                       (28.356406629821159, 281.24754646913954, 1146.6644894516851, 191.1107482419475)))

    nt.ok_(np.allclose(_stats(features_get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
                       (271.94122143951864, 271.94122143951864, 271.94122143951864, 271.94122143951864)))


def test_neurite_density_nrn():

    feat = 'neurite_volume_density'

    nt.ok_(np.allclose(_stats(features_get(feat, NRN)),
                       (0.24068543213643726, 0.52464681266899216, 1.4657913638494682, 0.36644784096236704)))

    nt.ok_(np.allclose(_stats(features_get(feat, NRN, neurite_type=NeuriteType.axon)),
                       (0.26289304906104355, 0.26289304906104355, 0.26289304906104355, 0.26289304906104355)))

    nt.ok_(np.allclose(_stats(features_get(feat, NRN, neurite_type=NeuriteType.basal_dendrite)),
                       (0.24068543213643726, 0.52464681266899216, 0.76533224480542938, 0.38266612240271469)))

    nt.ok_(np.allclose(_stats(features_get(feat, NRN, neurite_type=NeuriteType.apical_dendrite)),
                       (0.43756606998299519, 0.43756606998299519, 0.43756606998299519, 0.43756606998299519)))


def test_neurite_density_pop():

    feat = 'neurite_volume_density'

    nt.ok_(np.allclose(_stats(features_get(feat, POP)),
                       (6.1847539631150784e-06, 0.52464681266899216, 1.9767794901940539, 0.19767794901940539)))

    nt.ok_(np.allclose(_stats(features_get(feat, POP, neurite_type=NeuriteType.axon)),
                       (6.1847539631150784e-06, 0.26465213325053372, 0.5275513670655404, 0.17585045568851346)))

    nt.ok_(np.allclose(_stats(features_get(feat, POP, neurite_type=NeuriteType.basal_dendrite)),
                       (0.00034968816544949771, 0.52464681266899216, 1.0116620531455183, 0.16861034219091972)))

    nt.ok_(np.allclose(_stats(features_get(feat, POP, neurite_type=NeuriteType.apical_dendrite)),
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

    nt.eq_(ref, features_get(feat, nrt).tolist())
    nt.eq_(ref, features_get(feat, nrn).tolist())
    nt.eq_(ref, features_get(feat, pop).tolist())


def test_neurite_features_accept_single_tree():

    features = FEATURES['NEURITE'].keys()

    for f in features:
        ret = features_get(f, NRN.neurites[0])
        nt.ok_(ret.dtype.kind in ('i', 'f'))
        nt.ok_(len(ret) or len(ret) == 0) #  make sure that len() resolves


def test_register_neurite_feature_nrns():

    def npts(neurite):
        return len(neurite.points)

    def vol(neurite):
        return neurite.volume

    features.register_neurite_feature('foo', npts)

    n_points_ref = [len(n.points) for n in iter_neurites(NRNS)]
    n_points = features.get('foo', NRNS)
    assert_items_equal(n_points, n_points_ref)

    # test neurite type filtering
    n_points_ref = [len(n.points) for n in iter_neurites(NRNS, filt=_is_type(NeuriteType.axon))]
    n_points = features.get('foo', NRNS, neurite_type=NeuriteType.axon)
    assert_items_equal(n_points, n_points_ref)

    features.register_neurite_feature('bar', vol)

    n_volume_ref = [n.volume for n in iter_neurites(NRNS)]
    n_volume = features.get('bar', NRNS)
    assert_items_equal(n_volume, n_volume_ref)

    # test neurite type filtering
    n_volume_ref = [n.volume for n in iter_neurites(NRNS, filt=_is_type(NeuriteType.axon))]
    n_volume = features.get('bar', NRNS, neurite_type=NeuriteType.axon)
    assert_items_equal(n_volume, n_volume_ref)


def test_register_neurite_feature_pop():

    def npts(neurite):
        return len(neurite.points)

    def vol(neurite):
        return neurite.volume

    features.register_neurite_feature('foo', npts)

    n_points_ref = [len(n.points) for n in iter_neurites(POP)]
    n_points = features.get('foo', POP)
    assert_items_equal(n_points, n_points_ref)

    # test neurite type filtering
    n_points_ref = [len(n.points) for n in iter_neurites(POP,
                                                         filt=_is_type(NeuriteType.basal_dendrite))]
    n_points = features.get('foo', POP, neurite_type=NeuriteType.basal_dendrite)
    assert_items_equal(n_points, n_points_ref)

    features.register_neurite_feature('bar', vol)

    n_volume_ref = [n.volume for n in iter_neurites(POP)]
    n_volume = features.get('bar', POP)
    assert_items_equal(n_volume, n_volume_ref)

    # test neurite type filtering
    n_volume_ref = [n.volume for n in iter_neurites(POP, filt=_is_type(NeuriteType.basal_dendrite))]
    n_volume = features.get('bar', POP, neurite_type=NeuriteType.basal_dendrite)
    assert_items_equal(n_volume, n_volume_ref)


@nt.raises(NeuroMError)
def test_register_existing_feature_raises():
    features.register_neurite_feature('total_length', lambda n: None)


_PWD = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_PWD, '../../../test_data')

SWC_PATH = os.path.join(DATA_PATH, 'swc')
NEURON = load_neuron(os.path.join(SWC_PATH, 'Neuron.swc'))
SIMPLE = load_neuron(os.path.join(SWC_PATH, 'simple.swc'))


def test_section_lengths():
    ref_seclen = [n.length for n in iter_sections(NEURON)]
    seclen = features_get('section_lengths', NEURON)
    nt.eq_(len(seclen), 84)
    assert_allclose(seclen, ref_seclen)


def test_section_lengths_axon():
    s = features_get('section_lengths', NEURON, neurite_type=NeuriteType.axon)
    nt.eq_(len(s), 21)


def test_total_lengths_basal():
    s = features_get('section_lengths', NEURON, neurite_type=NeuriteType.basal_dendrite)
    nt.eq_(len(s), 42)


def test_section_lengths_apical():
    s = features_get('section_lengths', NEURON, neurite_type=NeuriteType.apical_dendrite)
    nt.eq_(len(s), 21)


def test_total_length_per_neurite_axon():
    tl = features_get('total_length_per_neurite', NEURON, neurite_type=NeuriteType.axon)
    nt.eq_(len(tl), 1)
    assert_allclose(tl, (207.87975221), rtol=1e-5)


def test_total_length_per_neurite_basal():
    tl = features_get('total_length_per_neurite', SIMPLE, neurite_type=NeuriteType.basal_dendrite)
    nt.eq_(len(tl), 1)
    assert_allclose(tl, (16,))


def test_total_length_per_neurite_apical():
    tl = features_get('total_length_per_neurite', NEURON, neurite_type=NeuriteType.apical_dendrite)
    nt.eq_(len(tl), 1)
    assert_allclose(tl, (214.37304578))


def test_total_length_axon():
    tl = features_get('total_length', NEURON, neurite_type=NeuriteType.axon)
    nt.eq_(len(tl), 1)
    assert_allclose(tl, (207.87975221), rtol=1e-5)


def test_total_length_basal():
    tl = features_get('total_length', NEURON, neurite_type=NeuriteType.basal_dendrite)
    nt.eq_(len(tl), 1)
    assert_allclose(tl, (418.43241644))


def test_total_length_apical():
    tl = features_get('total_length', NEURON, neurite_type=NeuriteType.apical_dendrite)
    nt.eq_(len(tl), 1)
    assert_allclose(tl, (214.37304578))


def test_section_lengths_invalid():
    s = features_get('section_lengths', NEURON, neurite_type=NeuriteType.soma)
    nt.eq_(len(s), 0)
    s = features_get('section_lengths', NEURON, neurite_type=NeuriteType.undefined)
    nt.eq_(len(s), 0)


def test_section_path_distances_axon():
    path_lengths = features_get('section_path_distances', NEURON, neurite_type=NeuriteType.axon)
    nt.eq_(len(path_lengths), 21)


def test_segment_lengths():
    ref_seglen = nf.segment_lengths(NEURON)
    seglen = features_get('segment_lengths', NEURON)
    nt.eq_(len(seglen), 840)
    assert_allclose(seglen, ref_seglen)


def test_local_bifurcation_angles():
    ref_local_bifangles = list(nf.local_bifurcation_angles(NEURON))

    local_bifangles = features_get('local_bifurcation_angles', NEURON)
    nt.eq_(len(local_bifangles), 40)
    assert_allclose(local_bifangles, ref_local_bifangles)

    s = features_get('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.axon)
    nt.eq_(len(s), 10)

    s = features_get('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.basal_dendrite)
    nt.eq_(len(s), 20)

    s = features_get('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.apical_dendrite)
    nt.eq_(len(s), 10)


def test_local_bifurcation_angles_invalid():
    s = features_get('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.soma)
    nt.eq_(len(s), 0)
    s = features_get('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.undefined)
    nt.eq_(len(s), 0)


def test_remote_bifurcation_angles():
    ref_remote_bifangles = list(nf.remote_bifurcation_angles(NEURON))
    remote_bifangles = features_get('remote_bifurcation_angles', NEURON)
    nt.eq_(len(remote_bifangles), 40)
    assert_allclose(remote_bifangles, ref_remote_bifangles)

    s = features_get('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.axon)
    nt.eq_(len(s), 10)

    s = features_get('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.basal_dendrite)
    nt.eq_(len(s), 20)

    s = features_get('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.apical_dendrite)
    nt.eq_(len(s), 10)


def test_remote_bifurcation_angles_invalid():
    s = features_get('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.soma)
    nt.eq_(len(s), 0)
    s = features_get('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.undefined)
    nt.eq_(len(s), 0)


def test_segment_radial_distances_origin():
    origin = (-100, -200, -300)
    ref_segs = nf.segment_radial_distances(NEURON)
    ref_segs_origin = nf.segment_radial_distances(NEURON, origin=origin)

    rad_dists = features_get('segment_radial_distances', NEURON)
    rad_dists_origin = features_get('segment_radial_distances', NEURON, origin=origin)

    nt.ok_(np.all(rad_dists == ref_segs))
    nt.ok_(np.all(rad_dists_origin == ref_segs_origin))
    nt.ok_(np.all(rad_dists_origin != ref_segs))

    nrns = [nm.load_neuron(os.path.join(SWC_PATH, f)) for
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

    rad_dists = features_get('section_radial_distances', NEURON)

    nt.eq_(len(rad_dists), 84)
    nt.ok_(np.all(rad_dists == ref_sec_rad_dist))

    nrns = [nm.load_neuron(os.path.join(SWC_PATH, f)) for
            f in ('point_soma_single_neurite.swc', 'point_soma_single_neurite2.swc')]
    pop = Population(nrns)
    rad_dist_nrns = [nm.get('section_radial_distances', nrn) for nrn in nrns]
    rad_dist_pop = nm.get('section_radial_distances', pop)
    assert_items_equal(rad_dist_pop, rad_dist_nrns)


def test_section_radial_distances_origin():
    origin = (-100, -200, -300)
    ref_sec_rad_dist_origin = nf.section_radial_distances(NEURON, origin=origin)
    rad_dists = features_get('section_radial_distances', NEURON, origin=origin)
    nt.eq_(len(rad_dists), 84)
    nt.ok_(np.all(rad_dists == ref_sec_rad_dist_origin))


def test_section_radial_axon():
    rad_dists = features_get('section_radial_distances', NEURON, neurite_type=NeuriteType.axon)
    nt.eq_(len(rad_dists), 21)


def test_number_of_sections_all():
    nt.eq_(features_get('number_of_sections', NEURON)[0], 84)


def test_number_of_sections_axon():
    nt.eq_(features_get('number_of_sections', NEURON, neurite_type=NeuriteType.axon)[0], 21)


def test_number_of_sections_basal():
    nt.eq_(features_get('number_of_sections', NEURON, neurite_type=NeuriteType.basal_dendrite)[0], 42)


def test_n_sections_apical():
    nt.eq_(features_get('number_of_sections', NEURON, neurite_type=NeuriteType.apical_dendrite)[0], 21)


def test_section_number_invalid():
    nt.eq_(features_get('number_of_sections', NEURON, neurite_type=NeuriteType.soma)[0], 0)
    nt.eq_(features_get('number_of_sections', NEURON, neurite_type=NeuriteType.undefined)[0], 0)


def test_per_neurite_number_of_sections():
    nsecs = features_get('number_of_sections_per_neurite', NEURON)
    nt.eq_(len(nsecs), 4)
    nt.ok_(np.all(nsecs == [21, 21, 21, 21]))


def test_per_neurite_number_of_sections_axon():
    nsecs = features_get('number_of_sections_per_neurite', NEURON, neurite_type=NeuriteType.axon)
    nt.eq_(len(nsecs), 1)
    nt.eq_(nsecs, [21])


def test_n_sections_per_neurite_basal():
    nsecs = features_get('number_of_sections_per_neurite', NEURON, neurite_type=NeuriteType.basal_dendrite)
    nt.eq_(len(nsecs), 2)
    nt.ok_(np.all(nsecs == [21, 21]))


def test_n_sections_per_neurite_apical():
    nsecs = features_get('number_of_sections_per_neurite', NEURON, neurite_type=NeuriteType.apical_dendrite)
    nt.eq_(len(nsecs), 1)
    nt.ok_(np.all(nsecs == [21]))


def test_neurite_number():
    nt.eq_(features_get('number_of_neurites', NEURON)[0], 4)
    nt.eq_(features_get('number_of_neurites', NEURON, neurite_type=NeuriteType.axon)[0], 1)
    nt.eq_(features_get('number_of_neurites', NEURON, neurite_type=NeuriteType.basal_dendrite)[0], 2)
    nt.eq_(features_get('number_of_neurites', NEURON, neurite_type=NeuriteType.apical_dendrite)[0], 1)
    nt.eq_(features_get('number_of_neurites', NEURON, neurite_type=NeuriteType.soma)[0], 0)
    nt.eq_(features_get('number_of_neurites', NEURON, neurite_type=NeuriteType.undefined)[0], 0)


def test_trunk_origin_radii():
    assert_allclose(features_get('trunk_origin_radii', NEURON),
                    [0.85351288499400002,
                     0.18391483031299999,
                     0.66943255462899998,
                     0.14656092843999999])

    assert_allclose(features_get('trunk_origin_radii', NEURON, neurite_type=NeuriteType.apical_dendrite),
                    [0.14656092843999999])
    assert_allclose(features_get('trunk_origin_radii', NEURON, neurite_type=NeuriteType.basal_dendrite),
                    [0.18391483031299999,
                     0.66943255462899998])
    assert_allclose(features_get('trunk_origin_radii', NEURON, neurite_type=NeuriteType.axon),
                    [0.85351288499400002])


def test_get_trunk_section_lengths():
    assert_allclose(features_get('trunk_section_lengths', NEURON),
                    [9.579117366740002,
                     7.972322416776259,
                     8.2245287740603779,
                     9.212707985134525])
    assert_allclose(features_get('trunk_section_lengths', NEURON, neurite_type=NeuriteType.apical_dendrite),
                    [9.212707985134525])
    assert_allclose(features_get('trunk_section_lengths', NEURON, neurite_type=NeuriteType.basal_dendrite),
                    [7.972322416776259, 8.2245287740603779])
    assert_allclose(features_get('trunk_section_lengths', NEURON, neurite_type=NeuriteType.axon),
                    [9.579117366740002])


def test_soma_radii():
    nt.assert_almost_equals(features_get('soma_radii', NEURON)[0], 0.130656, places=6)


def test_soma_surface_areas():
    area = 4. * math.pi * features_get('soma_radii', NEURON)[0] ** 2
    nt.eq_(features_get('soma_surface_areas', NEURON), area)


def test_sholl_frequency():
    assert_allclose(features_get('sholl_frequency', NEURON),
                    [4, 8, 8, 14, 9, 8, 7, 7])

    assert_allclose(features_get('sholl_frequency', NEURON, neurite_type=NeuriteType.apical_dendrite),
                    [1, 2, 2, 2, 2, 2, 1, 1])

    assert_allclose(features_get('sholl_frequency', NEURON, neurite_type=NeuriteType.basal_dendrite),
                    [2, 4, 4, 6, 5, 4, 4, 4])

    assert_allclose(features_get('sholl_frequency', NEURON, neurite_type=NeuriteType.axon),
                    [1, 2, 2, 6, 2, 2, 2, 2])


@nt.nottest  # test_get_segment_lengths is disabled in test_get_features
def test_section_path_distances_endpoint():

    ref_sec_path_len_start = list(iter_neurites(NEURON, sec.start_point_path_length))
    ref_sec_path_len = list(iter_neurites(NEURON, sec.end_point_path_length))
    path_lengths = features_get('section_path_distances', NEURON)
    nt.ok_(ref_sec_path_len != ref_sec_path_len_start)
    nt.eq_(len(path_lengths), 84)
    nt.ok_(np.all(path_lengths == ref_sec_path_len))

@nt.nottest  # test_get_segment_lengths is disabled in test_get_features
def test_section_path_distances_start_point():

    ref_sec_path_len_start = list(iter_neurites(NEURON, sec.start_point_path_length))
    path_lengths = features_get('section_path_distances', NEURON, use_start_point=True)
    nt.eq_(len(path_lengths), 84)
    nt.ok_(np.all(path_lengths == ref_sec_path_len_start))

def test_partition():
    nt.ok_(np.all(features_get('partition', NRNS)[:10] ==
                  np.array([ 19.,  17.,  15.,  13.,  11.,   9.,   7.,   5.,   3.,   1.])))

def test_partition_asymmetry():
    nt.ok_(np.allclose(features_get('partition_asymmetry', NRNS)[:10], np.array([0.9, 0.88888889, 0.875,
                                                                            0.85714286, 0.83333333,
                                                                            0.8, 0.75,  0.66666667,
                                                                            0.5,  0.])))

class MockNeuron:
   pass


def test_trunk_origin_elevations():
   n0 = load_neuron(('swc',"""
   1 1 0 0 0 4 -1
   2 3 1 0 0 2 1
   3 3 0 1 0 2 1
   """))

   n1 = load_neuron(('swc',"""
   1 1 0 0 0 4 -1
   2 3 0 -1 0 2 1
   """))

   pop = [n0, n1]
   assert_array_equal(features_get('trunk_origin_elevations', pop),
                      np.array([0.0, np.pi/2., -np.pi/2.], dtype=np.float32))
   nt.eq_(len(features_get('trunk_origin_elevations', pop, neurite_type=NeuriteType.axon)), 0)
#
#def test_trunk_origin_azimuths():
#    n0 = MockNeuron()
#    n1 = MockNeuron()
#    n2 = MockNeuron()
#    n3 = MockNeuron()
#    n4 = MockNeuron()
#    n5 = MockNeuron()
#
#    t = PointTree((0, 0, 0, 2))
#    t.type = NeuriteType.basal_dendrite
#    n0.neurites = [t]
#    n1.neurites = [t]
#    n2.neurites = [t]
#    n3.neurites = [t]
#    n4.neurites = [t]
#    n5.neurites = [t]
#    pop = [n0, n1, n2, n3, n4, n5]
#    s0 = make_soma([[0, 0, 1, 4]])
#    s1 = make_soma([[0, 0, -1, 4]])
#    s2 = make_soma([[0, 0, 0, 4]])
#    s3 = make_soma([[-1, 0, -1, 4]])
#    s4 = make_soma([[-1, 0, 0, 4]])
#    s5 = make_soma([[1, 0, 0, 4]])
#
#    pop[0].soma = s0
#    pop[1].soma = s1
#    pop[2].soma = s2
#    pop[3].soma = s3
#    pop[4].soma = s4
#    pop[5].soma = s5
#    nt.ok_(np.all(features_get('trunk_origin_azimuths', pop) ==
#                          [-np.pi/2., np.pi/2., 0.0, np.pi/4., 0.0, np.pi]))
#    nt.eq_(len(features_get('trunk_origin_azimuths', pop, neurite_type=NeuriteType.axon)), 0)

#def test_principal_directions_extents():
#    points = np.array([[-10., 0., 0.],
#                       [-9., 0., 0.],
#                       [9., 0., 0.],
#                       [10., 0., 0.]])
#
#    tree = PointTree(np.array([points[0][0], points[0][1], points[0][2], 1., 0., 0.]))
#    tree.add_child(PointTree(np.array([points[1][0], points[1][1], points[1][2], 1., 0., 0.])))
#    tree.children[0].add_child(PointTree(np.array([points[2][0], points[2][1], points[2][2], 1., 0., 0.])))
#    tree.children[0].add_child(PointTree(np.array([points[3][0], points[3][1], points[3][2], 1., 0., 0.])))
#
#    neurites = [tree, tree, tree]
#    extents0 = features_get('principal_direction_extents', neurites, direction='first')
#    nt.ok_(np.allclose(extents0, [20., 20., 20.]))
#    extents1 = features_get('principal_direction_extents', neurites, direction='second')
#    nt.ok_(np.allclose(extents1, [0., 0., 0.]))
#    extents2 = features_get('principal_direction_extents', neurites, direction='third')
#    nt.ok_(np.allclose(extents2, [0., 0., 0.]))

def test_get_doc():
    # test that no filtering returns more results than filter on 'radii'
    nt.ok_(len(_get_doc('')) > len(_get_doc('radii')))

    # test that filter on 'radii' returns more result than filter on 'alsjkdfas'
    nt.ok_(len(_get_doc('radii')) > len(_get_doc('alsjkdfas')))
