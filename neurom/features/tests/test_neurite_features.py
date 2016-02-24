
import os
from nose import tools as nt
import numpy as np
from neurom.core.tree import Tree
from neurom.core.types import TreeType
from neurom.features import neurite_features as nf
import neurom.sections as sec
import neurom.segments as seg
import neurom.bifurcations as bifs
from neurom import iter_neurites
from neurom.ezy import load_neuron

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')

NEURON_PATH = os.path.join(SWC_PATH, 'Neuron.swc')
NEURON = load_neuron(NEURON_PATH)

def test_section_lengths():

    ref_seclen = list(iter_neurites(NEURON, sec.length))
    seclen = list(nf.section_lengths(NEURON))
    nt.assert_equal(len(seclen), 84)
    nt.assert_true(np.all(seclen == ref_seclen))

    seclen = list(nf.section_lengths(NEURON, neurite_type=TreeType.all))
    nt.assert_equal(len(seclen), 84)
    nt.assert_true(np.all(seclen == ref_seclen))

def test_section_lengths_axon():
    s = list(nf.section_lengths(NEURON, neurite_type=TreeType.axon))
    nt.assert_equal(len(s), 21)

def test_section_lengths_basal():
    s = list(nf.section_lengths(NEURON, neurite_type=TreeType.basal_dendrite))
    nt.assert_equal(len(s), 42)

def test_section_lengths_apical():
    s = list(nf.section_lengths(NEURON, neurite_type=TreeType.apical_dendrite))
    nt.assert_equal(len(s), 21)

def test_section_lengths_invalid():
    s = list(nf.section_lengths(NEURON, neurite_type=TreeType.soma))
    nt.assert_equal(len(s), 0)
    s = nf.section_lengths(NEURON, neurite_type=TreeType.undefined)
    s = nf.section_lengths(NEURON, neurite_type=TreeType.soma)


def test_section_path_distances_endpoint():

    ref_sec_path_len_start = list(iter_neurites(NEURON, sec.start_point_path_length))
    ref_sec_path_len = list(iter_neurites(NEURON, sec.end_point_path_length))
    path_lengths = list(nf.section_path_distances(NEURON))
    nt.assert_true(ref_sec_path_len != ref_sec_path_len_start)
    nt.assert_equal(len(path_lengths), 84)
    nt.assert_true(np.all(path_lengths == ref_sec_path_len))

def test_section_path_distances_start_point():

    ref_sec_path_len_start = list(iter_neurites(NEURON, sec.start_point_path_length))
    path_lengths = list(nf.section_path_distances(NEURON, use_start_point=True))
    nt.assert_equal(len(path_lengths), 84)
    nt.assert_true(np.all(path_lengths == ref_sec_path_len_start))

def test_section_path_distances_axon():
    path_lengths = list(nf.section_path_distances(NEURON, neurite_type=TreeType.axon))
    nt.assert_equal(len(path_lengths), 21)

def test_segment_lengths():
    ref_seglen = list(iter_neurites(NEURON, seg.length))
    seglen = list(nf.segment_lengths(NEURON))
    nt.assert_equal(len(seglen), 840)
    nt.assert_true(np.all(seglen == ref_seglen))

    seglen = list(nf.segment_lengths(NEURON, neurite_type=TreeType.all))
    nt.assert_equal(len(seglen), 840)
    nt.assert_true(np.all(seglen == ref_seglen))

def test_local_bifurcation_angles():

    ref_local_bifangles = list(iter_neurites(NEURON, bifs.local_angle))

    local_bifangles = list(nf.local_bifurcation_angles(NEURON))
    nt.assert_equal(len(local_bifangles), 40)
    nt.assert_true(np.all(local_bifangles == ref_local_bifangles))
    local_bifangles = list(nf.local_bifurcation_angles(NEURON, neurite_type=TreeType.all))
    nt.assert_equal(len(local_bifangles), 40)
    nt.assert_true(np.all(local_bifangles == ref_local_bifangles))

def test_local_bifurcation_angles_axon():
    s = list(nf.local_bifurcation_angles(NEURON, neurite_type=TreeType.axon))
    nt.assert_equal(len(s), 10)

def test_local_bifurcation_angles_basal():
    s = list(nf.local_bifurcation_angles(NEURON, neurite_type=TreeType.basal_dendrite))
    nt.assert_equal(len(s), 20)

def test_local_bifurcation_angles_apical():
    s = list(nf.local_bifurcation_angles(NEURON, neurite_type=TreeType.apical_dendrite))
    nt.assert_equal(len(s), 10)

def test_local_bifurcation_angles_invalid():
    s = list(nf.local_bifurcation_angles(NEURON, neurite_type=TreeType.soma))
    nt.assert_equal(len(s), 0)
    s = list(nf.local_bifurcation_angles(NEURON, neurite_type=TreeType.undefined))
    nt.assert_equal(len(s), 0)

def test_remote_bifurcation_angles():
    ref_remote_bifangles = list(iter_neurites(NEURON, bifs.remote_angle))
    remote_bifangles = list(nf.remote_bifurcation_angles(NEURON))
    nt.assert_equal(len(remote_bifangles), 40)
    nt.assert_true(np.all(remote_bifangles == ref_remote_bifangles))
    remote_bifangles = list(nf.remote_bifurcation_angles(NEURON, neurite_type=TreeType.all))
    nt.assert_equal(len(remote_bifangles), 40)
    nt.assert_true(np.all(remote_bifangles == ref_remote_bifangles))

def test_remote_bifurcation_angles_axon():
    s = list(nf.remote_bifurcation_angles(NEURON, neurite_type=TreeType.axon))
    nt.assert_equal(len(s), 10)

def test_remote_bifurcation_angles_basal():
    s = list(nf.remote_bifurcation_angles(NEURON, neurite_type=TreeType.basal_dendrite))
    nt.assert_equal(len(s), 20)

def test_remote_bifurcation_angles_apical():
    s = list(nf.remote_bifurcation_angles(NEURON, neurite_type=TreeType.apical_dendrite))
    nt.assert_equal(len(s), 10)

def test_remote_bifurcation_angles_invalid():
    s = list(nf.remote_bifurcation_angles(NEURON, neurite_type=TreeType.soma))
    nt.assert_equal(len(s), 0)
    s = list(nf.remote_bifurcation_angles(NEURON, neurite_type=TreeType.undefined))
    nt.assert_equal(len(s), 0)

def test_section_radial_distances_endpoint():
    ref_sec_rad_dist_start = []
    for t in NEURON.neurites:
        ref_sec_rad_dist_start.extend(
            ll for ll in iter_neurites(t, sec.radial_dist(t.value, use_start_point=True)))

    ref_sec_rad_dist = []
    for t in NEURON.neurites:
        ref_sec_rad_dist.extend(ll for ll in iter_neurites(t, sec.radial_dist(t.value)))

    rad_dists = list(nf.section_radial_distances(NEURON))
    nt.assert_true(ref_sec_rad_dist != ref_sec_rad_dist_start)
    nt.assert_equal(len(rad_dists), 84)
    nt.assert_true(np.all(rad_dists == ref_sec_rad_dist))

def test_section_radial_distances_start_point():
    ref_sec_rad_dist_start = []
    for t in NEURON.neurites:
        ref_sec_rad_dist_start.extend(
            ll for ll in iter_neurites(t, sec.radial_dist(t.value, use_start_point=True)))

    rad_dists = list(nf.section_radial_distances(NEURON, use_start_point=True))
    nt.assert_equal(len(rad_dists), 84)
    nt.assert_true(np.all(rad_dists == ref_sec_rad_dist_start))

def test_section_radial_axon():
    rad_dists = list(nf.section_radial_distances(NEURON, neurite_type=TreeType.axon))
    nt.assert_equal(len(rad_dists), 21)

def test_section_number_all():
    nt.assert_equal(nf.section_number(NEURON).next(), 84)
    nt.assert_equal(nf.section_number(NEURON, neurite_type=TreeType.all).next(), 84)

def test_section_number_axon():
    nt.assert_equal(nf.section_number(NEURON, neurite_type=TreeType.axon).next(), 21)

def test_section_number_basal():
    nt.assert_equal(nf.section_number(NEURON, neurite_type=TreeType.basal_dendrite).next(), 42)

def test_n_sections_apical():
    nt.assert_equal(nf.section_number(NEURON, neurite_type=TreeType.apical_dendrite).next(), 21)

def test_seciton_number_invalid():
    nt.assert_equal(nf.section_number(NEURON, neurite_type=TreeType.soma).next(), 0)
    nt.assert_equal(nf.section_number(NEURON, neurite_type=TreeType.undefined).next(), 0)

def test_per_neurite_section_number():
    nsecs = list(nf.section_number_per_neurite(NEURON))
    nt.assert_equal(len(nsecs), 4)
    nt.assert_true(np.all(nsecs == [21, 21, 21, 21]))

def test_per_neurite_section_number_axon():
    nsecs = list(nf.section_number_per_neurite(NEURON, neurite_type=TreeType.axon))
    nt.assert_equal(len(nsecs), 1)
    nt.assert_equal(nsecs, [21])

def test_n_sections_per_neurite_basal():
    nsecs = list(nf.section_number_per_neurite(NEURON, neurite_type=TreeType.basal_dendrite))
    nt.assert_equal(len(nsecs), 2)
    nt.assert_true(np.all(nsecs == [21, 21]))

def test_n_sections_per_neurite_apical():
    nsecs = list(nf.section_number_per_neurite(NEURON, neurite_type=TreeType.apical_dendrite))
    nt.assert_equal(len(nsecs), 1)
    nt.assert_true(np.all(nsecs == [21]))

def test_neurite_number():
    nt.assert_equal(nf.neurite_number(NEURON).next(), 4)
    nt.assert_equal(nf.neurite_number(NEURON, neurite_type=TreeType.all).next(), 4)
    nt.assert_equal(nf.neurite_number(NEURON, neurite_type=TreeType.axon).next(), 1)
    nt.assert_equal(nf.neurite_number(NEURON, neurite_type=TreeType.basal_dendrite).next(), 2)
    nt.assert_equal(nf.neurite_number(NEURON, neurite_type=TreeType.apical_dendrite).next(), 1)
    nt.assert_equal(nf.neurite_number(NEURON, neurite_type=TreeType.soma).next(), 0)
    nt.assert_equal(nf.neurite_number(NEURON, neurite_type=TreeType.undefined).next(), 0)

def test_trunk_origin_radii():
    nt.assert_items_equal(list(nf.trunk_origin_radii(NEURON)),
                          [0.85351288499400002,
                           0.18391483031299999,
                           0.66943255462899998,
                           0.14656092843999999])

    nt.assert_items_equal(list(nf.trunk_origin_radii(NEURON, TreeType.apical_dendrite)),
                          [0.14656092843999999])
    nt.assert_items_equal(list(nf.trunk_origin_radii(NEURON, TreeType.basal_dendrite)),
                          [0.18391483031299999,
                           0.66943255462899998])
    nt.assert_items_equal(list(nf.trunk_origin_radii(NEURON, TreeType.axon)),
                          [0.85351288499400002])

def test_get_trunk_section_lengths():
    nt.assert_items_equal(nf.trunk_section_lengths(NEURON), [9.579117366740002,
                                                                   7.972322416776259,
                                                                   8.2245287740603779,
                                                                   9.212707985134525])
    nt.assert_items_equal(list(nf.trunk_section_lengths(NEURON, TreeType.apical_dendrite)), [9.212707985134525])
    nt.assert_items_equal(list(nf.trunk_section_lengths(NEURON, TreeType.basal_dendrite)),
                          [7.972322416776259, 8.2245287740603779])
    nt.assert_items_equal(list(nf.trunk_section_lengths(NEURON, TreeType.axon)), [9.579117366740002])

def test_principal_directions_extents():
    points = np.array([[-10., 0., 0.],
                    [-9., 0., 0.],
                    [9., 0., 0.],
                    [10., 0., 0.]])

    tree = Tree(np.array([points[0][0], points[0][1], points[0][2], 1., 0., 0.]))
    tree.add_child(Tree(np.array([points[1][0], points[1][1], points[1][2], 1., 0., 0.])))
    tree.children[0].add_child(Tree(np.array([points[2][0], points[2][1], points[2][2], 1., 0., 0.])))
    tree.children[0].add_child(Tree(np.array([points[3][0], points[3][1], points[3][2], 1., 0., 0.])))

    neurites = [tree, tree, tree]
    extents0 = list(nf.principal_directions_extents(neurites, direction='first'))
    nt.assert_true(np.allclose(extents0, [20., 20., 20.]))
    extents1 = list(nf.principal_directions_extents(neurites, direction='second'))
    nt.assert_true(np.allclose(extents1, [0., 0., 0.]))
    extents2 = list(nf.principal_directions_extents(neurites, direction='third'))
    nt.assert_true(np.allclose(extents2, [0., 0., 0.]))
