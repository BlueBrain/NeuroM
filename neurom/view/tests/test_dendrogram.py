import os
from numpy.testing import assert_array_almost_equal
import numpy as np
from nose import tools as nt
from neurom.core.types import NeuriteType
import neurom.view._dendrogram as dm
from neurom import load_neuron, get

_PWD = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_PWD, '../../../test_data/h5/v1/Neuron.h5')
NEURON = load_neuron(DATA_PATH)
NEURITE = NEURON.neurites[0]
TREE = NEURITE.root_node

OLD_OFFS = [1.2, -1.2]
NEW_OFFS = [2.3, -2.3]
SPACING = (40., 0.)

def test_n_rectangles_tree():

    nt.assert_equal(dm._n_rectangles(NEURITE), 230)


def test_n_rectangles_neuron():

    nt.assert_equal(dm._n_rectangles(NEURON), 920)


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

    xoffset = 100.
    xspace = 40.
    max_dims = [10., 2.]

    spx = dm._spacingx(TREE, max_dims, xoffset, xspace)

    nt.assert_almost_equal(spx, -120.)
    nt.assert_almost_equal(max_dims[0], 440.)


def test_update_offsets():

    start_x = -10.
    length = 44.

    offs = dm._update_offsets(start_x, SPACING, 2, OLD_OFFS, length)

    nt.assert_almost_equal(offs[0], 30.)
    nt.assert_almost_equal(offs[1], 42.8)


class TestDendrogram(object):

    def setUp(self):

        self.dtr = dm.Dendrogram(NEURITE)
        self.dnrn = dm.Dendrogram(NEURON)

        self.dtr.generate()
        self.dnrn.generate()

    def test_init(self):

        nt.assert_true(np.allclose(self.dnrn._rectangles.shape, (920, 4, 2)))

    def test_generate_tree(self):

        nt.assert_true(np.allclose(self.dtr._rectangles.shape, (230, 4, 2)))
        nt.assert_false(np.all(self.dtr._rectangles == 0.))

    def test_generate_soma(self):

        vrec = self.dnrn.soma
        assert_array_almost_equal(vrec,
                                  np.array([[-0.092495, -0.18499],
                                            [-0.092495,  0.],
                                            [0.092495,  0.],
                                            [0.092495, -0.18499]]))

        vrec = self.dtr.soma

        nt.assert_true(vrec == None)

    def test_neuron_not_corrupted(self):
        # Regression for #492: dendrogram was corrupting
        # neuron used to construct it.
        # This caused the section path distance calculation
        # to raise a KeyError exception.
        get('section_path_distances', NEURON)

    def test_generate_neuron(self):

        total = 0

        for n0, n1 in self.dnrn._groups:

            group = self.dnrn._rectangles[n0: n1]

            total += group.shape[0]

            nt.assert_false(np.all(group == 0.))

        nt.assert_equal(total, 920)

    def test_data(self):

        nt.assert_false(np.all(self.dnrn.data == 0.))
        nt.assert_false(np.all(self.dtr.data == 0.))

    def test_groups(self):
        nt.ok_(self.dnrn.groups)
        nt.ok_(self.dtr.groups)

    def test_dims(self):

        nt.ok_(self.dnrn.dims)
        nt.ok_(self.dtr.dims)

    def test_types_tree(self):
        for ctype in self.dtr.types:
            nt.eq_(ctype, NeuriteType.apical_dendrite)

    def test_types_neuron(self):
        types = tuple(self.dnrn.types)
        nt.eq_(types[0], NeuriteType.apical_dendrite)
        nt.eq_(types[1], NeuriteType.basal_dendrite)
