
from neurom.core.tree import Tree
from neurom.core.neuron import Neuron

import neurom.analysis.dendrogram as dm

TREE = Tree(np.array([0., 0., 0., 10., 1., 0., 0.]))

TREE.add_child(p.array([1., 1., 1., 5., 1., 0., 0.]))
TREE.add_child(p.array([1., 1., 1., 7., 1., 0., 0.]))

SOMA = make_soma(np.array([[0., 0., 0., 1., 1., 1., -1.]]))   
NEURON = Neuron(SOMA, [TREE, TREE, TREE])


def test_total_rectangles():

	nt.assert_equal(dm._total_rectangles(NEURON), 6)

class TestDendrogram(object):

	def setUp(self):

		self.dendro = Dendrogram(NEURON)

