import os
import math
import numpy as np
from nose import tools as nt
from neurom.features import neuron_features as nf
from neurom.ezy import load_neuron

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')

NEURON_PATH = os.path.join(SWC_PATH, 'Neuron.swc')
NEURON = load_neuron(NEURON_PATH)

def test_soma_radius():
    nt.assert_almost_equal(nf.soma_radius(NEURON).next(), 0.170710678)

def test_soma_surface_area():
    area = 4. * math.pi * (nf.soma_radius(NEURON).next() ** 2)
    nt.assert_almost_equal(nf.soma_surface_area(NEURON).next(), area)