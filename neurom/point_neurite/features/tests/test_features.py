import os
import math
from functools import partial
from nose import tools as nt
import numpy as np
from neurom.point_neurite.point_tree import PointTree
from neurom.core.soma import make_soma
from neurom.core.types import NeuriteType
import neurom.point_neurite.sections as sec
import neurom.point_neurite.segments as seg
import neurom.point_neurite.bifurcations as bifs
from neurom.point_neurite.features import get as get_feat
from neurom import iter_neurites
from neurom.point_neurite.io.utils import load_neuron as _load
from neurom.point_neurite.treefunc import set_tree_type as _set_tt

load_neuron = partial(_load, tree_action=_set_tt)

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')

NEURON_PATH = os.path.join(SWC_PATH, 'Neuron.swc')
NEURON = load_neuron(NEURON_PATH)
NEURONS = [NEURON, NEURON]


class MockNeuron:
    pass


def test_section_lengths():

    ref_seclen = list(iter_neurites(NEURON, sec.length))
    seclen = get_feat('section_lengths', NEURON)
    nt.assert_equal(len(seclen), 84)
    nt.assert_true(np.all(seclen == ref_seclen))

    seclen = get_feat('section_lengths', NEURON, neurite_type=NeuriteType.all)
    nt.assert_equal(len(seclen), 84)
    nt.assert_true(np.all(seclen == ref_seclen))


def test_section_lengths_axon():
    s = get_feat('section_lengths', NEURON, neurite_type=NeuriteType.axon)
    nt.assert_equal(len(s), 21)


def test_total_lengths_basal():
    s = get_feat('section_lengths', NEURON, neurite_type=NeuriteType.basal_dendrite)
    nt.assert_equal(len(s), 42)


def test_section_lengths_apical():
    s = get_feat('section_lengths', NEURON, neurite_type=NeuriteType.apical_dendrite)
    nt.assert_equal(len(s), 21)


def test_total_length_per_neurite_axon():
    tl = get_feat('total_length_per_neurite', NEURON, neurite_type=NeuriteType.axon)
    nt.assert_equal(len(tl), 1)
    nt.assert_true(np.allclose(tl, (207.87975221)))


def test_total_length_per_neurite_basal():
    tl = get_feat('total_length_per_neurite', NEURON, neurite_type=NeuriteType.basal_dendrite)
    nt.assert_equal(len(tl), 2)
    nt.assert_true(np.allclose(tl, (211.11737442, 207.31504202)))


def test_total_length_per_neurite_apical():
    tl = get_feat('total_length_per_neurite', NEURON, neurite_type=NeuriteType.apical_dendrite)
    nt.assert_equal(len(tl), 1)
    nt.assert_true(np.allclose(tl, (214.37304578)))


def test_total_length_axon():
    tl = get_feat('total_length', NEURON, neurite_type=NeuriteType.axon)
    nt.assert_equal(len(tl), 1)
    nt.assert_true(np.allclose(tl, (207.87975221)))


def test_total_length_basal():
    tl = get_feat('total_length', NEURON, neurite_type=NeuriteType.basal_dendrite)
    nt.assert_equal(len(tl), 1)
    nt.assert_true(np.allclose(tl, (418.43241644)))


def test_total_length_apical():
    tl = get_feat('total_length', NEURON, neurite_type=NeuriteType.apical_dendrite)
    nt.assert_equal(len(tl), 1)
    nt.assert_true(np.allclose(tl, (214.37304578)))


def test_section_lengths_invalid():
    s = get_feat('section_lengths', NEURON, neurite_type=NeuriteType.soma)
    nt.assert_equal(len(s), 0)
    s = get_feat('section_lengths', NEURON, neurite_type=NeuriteType.undefined)
    nt.assert_equal(len(s), 0)


def test_section_path_distances_endpoint():

    ref_sec_path_len_start = list(iter_neurites(NEURON, sec.start_point_path_length))
    ref_sec_path_len = list(iter_neurites(NEURON, sec.end_point_path_length))
    path_lengths = get_feat('section_path_distances', NEURON)
    nt.assert_true(ref_sec_path_len != ref_sec_path_len_start)
    nt.assert_equal(len(path_lengths), 84)
    nt.assert_true(np.all(path_lengths == ref_sec_path_len))


def test_section_path_distances_start_point():

    ref_sec_path_len_start = list(iter_neurites(NEURON, sec.start_point_path_length))
    path_lengths = get_feat('section_path_distances', NEURON, use_start_point=True)
    nt.assert_equal(len(path_lengths), 84)
    nt.assert_true(np.all(path_lengths == ref_sec_path_len_start))


def test_section_path_distances_axon():
    path_lengths = get_feat('section_path_distances', NEURON, neurite_type=NeuriteType.axon)
    nt.assert_equal(len(path_lengths), 21)


def test_segment_lengths():
    ref_seglen = list(iter_neurites(NEURON, seg.length))
    seglen = get_feat('segment_lengths', NEURON)
    nt.assert_equal(len(seglen), 840)
    nt.assert_true(np.all(seglen == ref_seglen))

    seglen = get_feat('segment_lengths', NEURON, neurite_type=NeuriteType.all)
    nt.assert_equal(len(seglen), 840)
    nt.assert_true(np.all(seglen == ref_seglen))


def test_local_bifurcation_angles():

    ref_local_bifangles = list(iter_neurites(NEURON, bifs.local_angle))

    local_bifangles = get_feat('local_bifurcation_angles', NEURON)
    nt.assert_equal(len(local_bifangles), 40)
    nt.assert_true(np.all(local_bifangles == ref_local_bifangles))
    local_bifangles = get_feat('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.all)
    nt.assert_equal(len(local_bifangles), 40)
    nt.assert_true(np.all(local_bifangles == ref_local_bifangles))

def test_local_bifurcation_angles_axon():
    s = get_feat('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.axon)
    nt.assert_equal(len(s), 10)

def test_local_bifurcation_angles_basal():
    s = get_feat('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.basal_dendrite)
    nt.assert_equal(len(s), 20)


def test_local_bifurcation_angles_apical():
    s = get_feat('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.apical_dendrite)
    nt.assert_equal(len(s), 10)


def test_local_bifurcation_angles_invalid():
    s = get_feat('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.soma)
    nt.assert_equal(len(s), 0)
    s = get_feat('local_bifurcation_angles', NEURON, neurite_type=NeuriteType.undefined)
    nt.assert_equal(len(s), 0)


def test_remote_bifurcation_angles():
    ref_remote_bifangles = list(iter_neurites(NEURON, bifs.remote_angle))
    remote_bifangles = get_feat('remote_bifurcation_angles', NEURON)
    nt.assert_equal(len(remote_bifangles), 40)
    nt.assert_true(np.all(remote_bifangles == ref_remote_bifangles))
    remote_bifangles = get_feat('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.all)
    nt.assert_equal(len(remote_bifangles), 40)
    nt.assert_true(np.all(remote_bifangles == ref_remote_bifangles))


def test_remote_bifurcation_angles_axon():
    s = get_feat('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.axon)
    nt.assert_equal(len(s), 10)


def test_remote_bifurcation_angles_basal():
    s = get_feat('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.basal_dendrite)
    nt.assert_equal(len(s), 20)


def test_remote_bifurcation_angles_apical():
    s = get_feat('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.apical_dendrite)
    nt.assert_equal(len(s), 10)


def test_remote_bifurcation_angles_invalid():
    s = get_feat('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.soma)
    nt.assert_equal(len(s), 0)
    s = get_feat('remote_bifurcation_angles', NEURON, neurite_type=NeuriteType.undefined)
    nt.assert_equal(len(s), 0)


def test_segment_radial_distances():
    ref_segs = []
    for t in NEURON.neurites:
        ref_segs.extend(ll for ll in iter_neurites(t, seg.radial_dist(t.value)))

    rad_dists = get_feat('segment_radial_distances', NEURON)
    nt.assert_true(np.all(rad_dists == ref_segs))


def test_segment_radial_distances_origin():
    origin = (-100, -200, -300)

    ref_segs = []
    ref_segs2 = []
    for t in NEURON.neurites:
        ref_segs.extend(ll for ll in iter_neurites(t, seg.radial_dist(t.value)))
        ref_segs2.extend(ll for ll in iter_neurites(t, seg.radial_dist(origin)))

    rad_dists = get_feat('segment_radial_distances', NEURON)
    rad_dists2 = get_feat('segment_radial_distances', NEURON, origin=origin)
    nt.assert_true(np.all(rad_dists == ref_segs))
    nt.assert_true(np.all(rad_dists2 == ref_segs2))
    nt.assert_true(np.all(rad_dists2 != ref_segs))


def test_section_radial_distances_endpoint():
    ref_sec_rad_dist_start = []
    for t in NEURON.neurites:
        ref_sec_rad_dist_start.extend(
            ll for ll in iter_neurites(t, sec.radial_dist(t.value, use_start_point=True)))

    ref_sec_rad_dist = []
    for t in NEURON.neurites:
        ref_sec_rad_dist.extend(ll for ll in iter_neurites(t, sec.radial_dist(t.value)))

    rad_dists = get_feat('section_radial_distances', NEURON)
    nt.assert_true(ref_sec_rad_dist != ref_sec_rad_dist_start)
    nt.assert_equal(len(rad_dists), 84)
    nt.assert_true(np.all(rad_dists == ref_sec_rad_dist))


def test_section_radial_distances_start_point():
    ref_sec_rad_dist_start = []
    for t in NEURON.neurites:
        ref_sec_rad_dist_start.extend(
            ll for ll in iter_neurites(t, sec.radial_dist(t.value, use_start_point=True)))

    rad_dists = get_feat('section_radial_distances', NEURON, use_start_point=True)
    nt.assert_equal(len(rad_dists), 84)
    nt.assert_true(np.all(rad_dists == ref_sec_rad_dist_start))


def test_section_radial_axon():
    rad_dists = get_feat('section_radial_distances', NEURON, neurite_type=NeuriteType.axon)
    nt.assert_equal(len(rad_dists), 21)


def test_number_of_sections_all():
    nt.assert_equal(get_feat('number_of_sections', NEURON)[0], 84)
    nt.assert_equal(get_feat('number_of_sections', NEURON, neurite_type=NeuriteType.all)[0], 84)


def test_number_of_sections_axon():
    nt.assert_equal(get_feat('number_of_sections', NEURON, neurite_type=NeuriteType.axon)[0], 21)


def test_number_of_sections_basal():
    nt.assert_equal(get_feat('number_of_sections', NEURON, neurite_type=NeuriteType.basal_dendrite)[0], 42)


def test_n_sections_apical():
    nt.assert_equal(get_feat('number_of_sections', NEURON, neurite_type=NeuriteType.apical_dendrite)[0], 21)


def test_seciton_number_invalid():
    nt.assert_equal(get_feat('number_of_sections', NEURON, neurite_type=NeuriteType.soma)[0], 0)
    nt.assert_equal(get_feat('number_of_sections', NEURON, neurite_type=NeuriteType.undefined)[0], 0)


def test_per_neurite_number_of_sections():
    nsecs = get_feat('number_of_sections_per_neurite', NEURON)
    nt.assert_equal(len(nsecs), 4)
    nt.assert_true(np.all(nsecs == [21, 21, 21, 21]))


def test_per_neurite_number_of_sections_axon():
    nsecs = get_feat('number_of_sections_per_neurite', NEURON, neurite_type=NeuriteType.axon)
    nt.assert_equal(len(nsecs), 1)
    nt.assert_equal(nsecs, [21])


def test_n_sections_per_neurite_basal():
    nsecs = get_feat('number_of_sections_per_neurite', NEURON, neurite_type=NeuriteType.basal_dendrite)
    nt.assert_equal(len(nsecs), 2)
    nt.assert_true(np.all(nsecs == [21, 21]))


def test_n_sections_per_neurite_apical():
    nsecs = get_feat('number_of_sections_per_neurite', NEURON, neurite_type=NeuriteType.apical_dendrite)
    nt.assert_equal(len(nsecs), 1)
    nt.assert_true(np.all(nsecs == [21]))


def test_neurite_number():
    nt.assert_equal(get_feat('number_of_neurites', NEURON)[0], 4)
    nt.assert_equal(get_feat('number_of_neurites', NEURON, neurite_type=NeuriteType.all)[0], 4)
    nt.assert_equal(get_feat('number_of_neurites', NEURON, neurite_type=NeuriteType.axon)[0], 1)
    nt.assert_equal(get_feat('number_of_neurites', NEURON, neurite_type=NeuriteType.basal_dendrite)[0], 2)
    nt.assert_equal(get_feat('number_of_neurites', NEURON, neurite_type=NeuriteType.apical_dendrite)[0], 1)
    nt.assert_equal(get_feat('number_of_neurites', NEURON, neurite_type=NeuriteType.soma)[0], 0)
    nt.assert_equal(get_feat('number_of_neurites', NEURON, neurite_type=NeuriteType.undefined)[0], 0)


def test_trunk_origin_radii():
    nt.assert_items_equal(get_feat('trunk_origin_radii', NEURON),
                          [0.85351288499400002,
                           0.18391483031299999,
                           0.66943255462899998,
                           0.14656092843999999])

    nt.assert_items_equal(get_feat('trunk_origin_radii', NEURON, NeuriteType.apical_dendrite),
                          [0.14656092843999999])
    nt.assert_items_equal(get_feat('trunk_origin_radii', NEURON, NeuriteType.basal_dendrite),
                          [0.18391483031299999,
                           0.66943255462899998])
    nt.assert_items_equal(get_feat('trunk_origin_radii', NEURON, NeuriteType.axon),
                          [0.85351288499400002])


def test_get_trunk_section_lengths():
    nt.assert_items_equal(get_feat('trunk_section_lengths', NEURON), [9.579117366740002,
                                                                      7.972322416776259,
                                                                      8.2245287740603779,
                                                                      9.212707985134525])
    nt.assert_items_equal(get_feat('trunk_section_lengths', NEURON, NeuriteType.apical_dendrite),
                          [9.212707985134525])
    nt.assert_items_equal(get_feat('trunk_section_lengths', NEURON, NeuriteType.basal_dendrite),
                          [7.972322416776259, 8.2245287740603779])
    nt.assert_items_equal(get_feat('trunk_section_lengths', NEURON, NeuriteType.axon), [9.579117366740002])


def test_principal_directions_extents():
    points = np.array([[-10., 0., 0.],
                       [-9., 0., 0.],
                       [9., 0., 0.],
                       [10., 0., 0.]])

    tree = PointTree(np.array([points[0][0], points[0][1], points[0][2], 1., 0., 0.]))
    tree.add_child(PointTree(np.array([points[1][0], points[1][1], points[1][2], 1., 0., 0.])))
    tree.children[0].add_child(PointTree(np.array([points[2][0], points[2][1], points[2][2], 1., 0., 0.])))
    tree.children[0].add_child(PointTree(np.array([points[3][0], points[3][1], points[3][2], 1., 0., 0.])))

    neurites = [tree, tree, tree]
    extents0 = get_feat('principal_direction_extents', neurites, direction='first')
    nt.assert_true(np.allclose(extents0, [20., 20., 20.]))
    extents1 = get_feat('principal_direction_extents', neurites, direction='second')
    nt.assert_true(np.allclose(extents1, [0., 0., 0.]))
    extents2 = get_feat('principal_direction_extents', neurites, direction='third')
    nt.assert_true(np.allclose(extents2, [0., 0., 0.]))


def test_soma_radii():
    nt.assert_true(np.all(get_feat('soma_radii', NEURONS) ==
                          [0.17071067811865476, 0.17071067811865476]))

def test_soma_surface_areas():
    area = 4. * math.pi * get_feat('soma_radii', NEURON)[0] ** 2
    nt.assert_true(np.all(get_feat('soma_surface_areas', NEURONS) == [area, area]))

def test_trunk_origin_elevations():
    n0 = MockNeuron()
    n1 = MockNeuron()

    s = make_soma([[0, 0, 0, 4]])
    t0 = PointTree((1, 0, 0, 2))
    t0.type = NeuriteType.basal_dendrite
    t1 = PointTree((0, 1, 0, 2))
    t1.type = NeuriteType.basal_dendrite
    n0.neurites = [t0, t1]
    n0.soma = s

    t2 = PointTree((0, -1, 0, 2))
    t2.type = NeuriteType.basal_dendrite
    n1.neurites = [t2]
    n1.soma = s

    pop = [n0, n1]
    nt.assert_true(np.all(get_feat('trunk_origin_elevations', pop) ==
                          [0.0, np.pi/2., -np.pi/2.]))
    nt.eq_(len(get_feat('trunk_origin_elevations', pop, neurite_type=NeuriteType.axon)), 0)

def test_trunk_origin_azimuths():
    n0 = MockNeuron()
    n1 = MockNeuron()
    n2 = MockNeuron()
    n3 = MockNeuron()
    n4 = MockNeuron()
    n5 = MockNeuron()

    t = PointTree((0, 0, 0, 2))
    t.type = NeuriteType.basal_dendrite
    n0.neurites = [t]
    n1.neurites = [t]
    n2.neurites = [t]
    n3.neurites = [t]
    n4.neurites = [t]
    n5.neurites = [t]
    pop = [n0, n1, n2, n3, n4, n5]
    s0 = make_soma([[0, 0, 1, 4]])
    s1 = make_soma([[0, 0, -1, 4]])
    s2 = make_soma([[0, 0, 0, 4]])
    s3 = make_soma([[-1, 0, -1, 4]])
    s4 = make_soma([[-1, 0, 0, 4]])
    s5 = make_soma([[1, 0, 0, 4]])

    pop[0].soma = s0
    pop[1].soma = s1
    pop[2].soma = s2
    pop[3].soma = s3
    pop[4].soma = s4
    pop[5].soma = s5
    nt.assert_true(np.all(get_feat('trunk_origin_azimuths', pop) ==
                          [-np.pi/2., np.pi/2., 0.0, np.pi/4., 0.0, np.pi]))
    nt.eq_(len(get_feat('trunk_origin_azimuths', pop, neurite_type=NeuriteType.axon)), 0)
