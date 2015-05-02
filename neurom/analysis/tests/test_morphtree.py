from nose import tools as nt
import os
from neurom.core.point import as_point
from neurom.core.tree import Tree
import neurom.core.tree as tr
from neurom.io.utils import make_neuron
from neurom.io.readers import load_data
from neurom.analysis.morphmath import point_dist
from neurom.analysis.morphtree import get_segment_lengths
from neurom.analysis.morphtree import get_segment_diameters
from neurom.analysis.morphtree import get_segment_radial_dists
from neurom.analysis.morphtree import get_segment_path_distance
from neurom.analysis.morphtree import find_tree_type
from neurom.analysis.morphtree import get_tree_type

DATA_PATH = './test_data'
SWC_PATH = os.path.join(DATA_PATH, 'swc/')

data    = load_data(SWC_PATH + 'Neuron.swc')
neuron0 = make_neuron(data)
tree_types = ['axon', 'basal', 'basal', 'apical']

def form_neuron_tree():
    p = [1, 1, 0.0, 0.0, 0.0, 1.0, 2]
    T = Tree(p)
    T1 = T.add_child(Tree([1, 1, 0.0, 1.0, 0.0, 1.0, 2]))
    T2 = T1.add_child(Tree([1, 1, 0.0, 2.0, 0.0, 1.0, 2]))
    T3 = T2.add_child(Tree([1, 1, 0.0, 4.0, 0.0, 2.0, 2]))
    T4 = T3.add_child(Tree([1, 1, 0.0, 5.0, 0.0, 2.0, 2]))
    T5 = T4.add_child(Tree([1, 1, 2.0, 5.0, 0.0, 1.0, 2]))
    T6 = T4.add_child(Tree([1, 1, 0.0, 5.0, 2.0, 1.0, 2]))
    T7 = T5.add_child(Tree([1, 1, 3.0, 5.0, 0.0, 0.75, 2]))
    T8 = T7.add_child(Tree([1, 1, 4.0, 5.0, 0.0, 0.75, 2]))
    T9 = T6.add_child(Tree([1, 1, 0.0, 5.0, 3.0, 0.75, 2]))
    T10 = T9.add_child(Tree([1, 1, 0.0, 6.0, 3.0, 0.75,2]))
    return T


def form_simple_tree():
    p = [1, 1, 0.0, 0.0, 0.0, 1.0, 1]
    T = Tree(p)
    T1 = T.add_child(Tree([1, 1, 0.0, 2.0, 0.0, 1.0, 1]))
    T2 = T1.add_child(Tree([1, 1, 0.0, 4.0, 0.0, 1.0, 1]))
    T3 = T2.add_child(Tree([1, 1, 0.0, 6.0, 0.0, 1.0, 1]))
    T4 = T3.add_child(Tree([1, 1, 0.0, 8.0, 0.0, 1.0, 1]))


    T5 = T.add_child(Tree([1, 1, 0.0, 0.0, 2.0, 1.0, 1]))
    T6 = T5.add_child(Tree([1, 1, 0.0, 0.0, 4.0, 1.0, 1]))
    T7 = T6.add_child(Tree([1, 1, 0.0, 0.0, 6.0, 1.0, 1]))
    T8 = T7.add_child(Tree([1, 1, 0.0, 0.0, 8.0, 1.0, 1]))

    return T


def test_segment_lengths():

    T = form_neuron_tree()

    lg = get_segment_lengths(T)

    nt.assert_equal(lg, [1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0])


def test_segment_diameters():

    T = form_neuron_tree()

    dia = get_segment_diameters(T)

    nt.assert_equal(dia, [2.0, 2.0, 3.0, 4.0, 3.0, 1.75, 1.5, 3.0, 1.75, 1.5])


def test_segment_radial_dist():
    T = form_simple_tree()

    p= [0.0, 0.0, 0.0]

    rd = get_segment_radial_dists(p,T)

    nt.assert_equal(rd, [1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0])


def test_segment_path_length():
    leaves = [l for l in tr.iter_leaf(form_neuron_tree())]
    for l in leaves:
        nt.ok_(get_segment_path_distance(l) == 9)


def test_find_tree_type():
    for en_tree, test_tree in enumerate(neuron0.neurite_trees):
        find_tree_type(test_tree)
        nt.ok_(test_tree.type == tree_types[en_tree])


def test_get_tree_type():
    for en_tree, test_tree in enumerate(neuron0.neurite_trees):
        if hasattr(test_tree, 'type'):
            del test_tree.type
        # tree.type should be computed here.
        nt.ok_(get_tree_type(test_tree) == tree_types[en_tree])
        test_tree = find_tree_type(test_tree)
        # tree.type should already exists here, from previous action.
        nt.ok_(get_tree_type(test_tree) == tree_types[en_tree])
