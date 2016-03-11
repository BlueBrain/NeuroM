import os
import math
import numpy as np
from nose import tools as nt
from neurom.core.tree import Tree
from neurom.core.neuron import make_soma
from neurom.features import neuron_features as nf
from neurom.core.types import TreeType
from neurom.ezy import load_neuron

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')

class MockNeuron: pass

NEURON_PATH = os.path.join(SWC_PATH, 'Neuron.swc')
NEURON = load_neuron(NEURON_PATH)
NEURONS = [NEURON, NEURON]

def test_soma_radii():
    nt.eq_(list(nf.soma_radii(NEURONS)), [0.17071067811865476, 0.17071067811865476])

def test_soma_surface_areas():
    area = 4. * math.pi * list(nf.soma_radii(NEURON))[0] ** 2
    nt.eq_(list(nf.soma_surface_areas(NEURONS)), [area, area])

def test_trunk_origin_elevations():
    n0 = MockNeuron()
    n1 = MockNeuron()
    n2 = MockNeuron()

    s = make_soma([[0, 0, 0, 4]])
    t0 = Tree((1, 0, 0, 2))
    t0.type = TreeType.basal_dendrite
    t1 = Tree((0, 1, 0, 2))
    t1.type = TreeType.basal_dendrite
    n0.neurites = [t0, t1]
    n0.soma = s

    t2 = Tree((0, -1, 0, 2))
    t2.type = TreeType.basal_dendrite
    n1.neurites = [t2]
    n1.soma = s

    pop = [n0, n1]
    nt.eq_(list(nf.trunk_origin_elevations(pop)), [0.0, np.pi/2., -np.pi/2.])
    nt.eq_(list(nf.trunk_origin_elevations(pop, neurite_type=TreeType.axon)), [])

def test_trunk_origin_azimuths():
    n0 = MockNeuron()
    n1 = MockNeuron()
    n2 = MockNeuron()
    n3 = MockNeuron()
    n4 = MockNeuron()
    n5 = MockNeuron()

    t = Tree((0, 0, 0, 2))
    t.type = TreeType.basal_dendrite
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
    nt.eq_(list(nf.trunk_origin_azimuths(pop)), [-np.pi/2., np.pi/2., 0.0, np.pi/4., 0.0, np.pi])
    nt.eq_(list(nf.trunk_origin_azimuths(pop, neurite_type=TreeType.axon)), [])
