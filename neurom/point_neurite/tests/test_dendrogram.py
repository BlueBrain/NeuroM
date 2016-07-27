
import numpy as np
from nose import tools as nt
from neurom.point_neurite.point_tree import PointTree
from neurom.point_neurite.core import Neuron
from neurom.core.soma import make_soma
from neurom.core.types import NeuriteType
from neurom.point_neurite.treefunc import set_tree_type
import neurom.point_neurite.dendrogram as dm


TREE = PointTree(np.array([0., 0., 0., 10., 4., 0., 0.]))
TREE.add_child(PointTree(np.array([3., 3., 3., 9., 4., 0., 0.])))

TREE.children[0].add_child(PointTree(np.array([10., 10., 10., 5., 4., 0., 0.])))
TREE.children[0].add_child(PointTree(np.array([-10., -10., -10., 7., 4., 0., 0.])))

set_tree_type(TREE)

SOMA = make_soma(np.array([[0., 0., 0., 1., 1., 1., -1.]]))
NEURON = Neuron(SOMA, [TREE, TREE, TREE])

OLD_OFFS = [1.2, -1.2]
NEW_OFFS = [2.3, -2.3]
SPACING = (40., 0.)

def test_n_rectangles_tree():

    nt.assert_equal(dm._n_rectangles(TREE), 5)


def test_n_rectangles_neuron():

    nt.assert_equal(dm._n_rectangles(NEURON), 16)


def test_n_rectangles_other():

    nt.assert_equal(dm._n_rectangles('I am unique.'), 0)


def test_vertical_segment():

    radii = [10., 20.]

    res = np.array([[ -7.7,  -1.2],
                    [-17.7,  -2.3],
                    [ 22.3,  -2.3],
                    [ 12.3,  -1.2]])

    seg = dm._vertical_segment(OLD_OFFS, NEW_OFFS, SPACING, radii)

    nt.assert_true(np.allclose(seg, res))


def test_horizontal_segment():

    diameter = 10.

    res = np.array([[  1.2,  -1.2],
                    [  2.3,  -1.2],
                    [  2.3, -11.2],
                    [  1.2, -11.2]])

    seg = dm._horizontal_segment(OLD_OFFS, NEW_OFFS, SPACING, diameter)

    nt.assert_true(np.allclose(seg, res))


def test_spacingx():

    xoffset = 100.2
    xspace = 40.
    max_dims = [10., 2.]

    spx = dm._spacingx(TREE, max_dims, xoffset, xspace)

    nt.assert_almost_equal(spx, 60.2)
    nt.assert_almost_equal(max_dims[0], 2. * xspace)


def test_update_offsets():

    start_x = -10.
    length = 44.

    offs = dm._update_offsets(start_x, SPACING, 2, OLD_OFFS, length)

    nt.assert_almost_equal(offs[0], 30.)
    nt.assert_almost_equal(offs[1], 42.8)

class TestDendrogram(object):

    def setUp(self):

        self.dtr = dm.Dendrogram(TREE)
        self.dnrn = dm.Dendrogram(NEURON)

        self.dtr.generate()
        self.dnrn.generate()

    def test_init(self):

        nt.assert_true(np.allclose(self.dnrn._rectangles.shape, (16, 4, 2)))

    def test_generate_tree(self):

        nt.assert_true(np.allclose(self.dtr._rectangles.shape, (5, 4, 2)))
        nt.assert_false(np.all(self.dtr._rectangles == 0.))

    def test_generate_neuron(self):

        total = 0

        for n0, n1 in self.dnrn._groups:

            group = self.dnrn._rectangles[n0: n1]

            total += group.shape[0]

            nt.assert_false(np.all(group == 0.))

        nt.assert_equal(total, 15)

    def test_data(self):

        nt.assert_false(np.all(self.dnrn.data == 0.))
        nt.assert_false(np.all(self.dtr.data == 0.))

    def test_groups(self):

        nt.assert_false(not self.dnrn.groups)
        nt.assert_false(not self.dtr.groups)

    def test_dims(self):

        nt.assert_false(not self.dnrn.dims)
        nt.assert_false(not self.dtr.dims)

    def test_types_tree(self):

        for ctype in self.dtr.types:
            print ctype
            nt.assert_true(ctype == NeuriteType.apical_dendrite)

    def test_types_neuron(self):

        for ctype in self.dnrn.types:
            print ctype
            nt.assert_true(ctype == NeuriteType.apical_dendrite)


    def test_generate_dendro(self):pass



