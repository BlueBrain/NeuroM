
import numpy as np
from nose import tools as nt
from neurom.core.tree import Tree
from neurom.core.neuron import Neuron, make_soma

import neurom.analysis.dendrogram as dm

TREE = Tree(np.array([0., 0., 0., 10., 1., 0., 0.]))
TREE.add_child(Tree(np.array([3., 3., 3., 9., 1., 0., 0.])))

TREE.children[0].add_child(Tree(np.array([10., 10., 10., 5., 1., 0., 0.])))
TREE.children[0].add_child(Tree(np.array([-10., -10., -10., 7., 1., 0., 0.])))

SOMA = make_soma(np.array([[0., 0., 0., 1., 1., 1., -1.]]))   
NEURON = Neuron(SOMA, [TREE, TREE, TREE])


def test_n_rectangles_tree():

	nt.assert_equal(dm._n_rectangles(TREE), 5)


def test_n_rectangles_neuron():

	nt.assert_equal(dm._n_rectangles(NEURON), 15)


def test_displace():

	rects = np.array([[[-0.85351288,  0.0],
				       [-0.72855526,  0.1],
				       [ 0.72855526,  0.1],
				       [ 0.85351288,  0.0]],
			          [[-0.72855526,  0.1],
			           [-0.54222909,  0.75137071],
			           [ 0.54222909,  0.75137071],
			           [ 0.72855526,  0.1]]])

	res = np.array([[[ 2.14648711,  -1.        ],
				        [ 2.27144473,  -0.9       ],
				        [ 3.72855526,  -0.9        ],
				        [ 3.85351288,  -1.        ]],
				       [[ 2.27144473,  -0.9       ],
				        [ 2.45777091,  -0.24862929],
				        [ 3.54222909,  -0.24862929],
				        [ 3.72855526,  -0.9       ]]])

	dm.displace(rects, (3., -1.))
	nt.assert_true(np.allclose(rects, res))


def test_vertical_segment():

	old_offs = np.array([1.2, -1.2])
	new_offs = np.array([2.3, -2.3])
	spacing = (40., 0.)
	radii = [10., 20.]

	res = np.array([[ -7.7,  -1.2],
       				[-17.7,  -2.3],
       				[ 22.3,  -2.3],
       				[ 12.3,  -1.2]])

	seg = dm._vertical_segment(old_offs, new_offs, spacing, radii)

	nt.assert_true(np.allclose(seg, res))


def test_horizontal_segment():

	old_offs = np.array([1.2, -1.2])
	new_offs = np.array([2.3, -2.3])
	spacing = (40., 0.)
	diameter = 10.

	res = np.array([[  1.2,  -1.2],
       				[  2.3,  -1.2],
       				[  2.3, -11.2],
       				[  1.2, -11.2]])

	seg = dm._horizontal_segment(old_offs, new_offs, spacing, diameter)

	nt.assert_true(np.allclose(seg, res))

def test_spacingx():pass
def test_update_offsets():pass

class TestDendrogram(object):

	def setUp(self):

		self.dendro = dm.Dendrogram(NEURON)

	def test_generate(self):pass

	def test_generate_dendro(self):pass

