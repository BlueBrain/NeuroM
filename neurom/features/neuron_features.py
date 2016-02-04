from itertools import product
from neurom.core.types import TreeType
from neurom.core.types import tree_type_checker
from neurom import segments as _seg
from neurom import sections as _sec
from neurom import bifurcations as _bifs
from neurom import points as _pts
from neurom import iter_neurites
from neurom.core.neuron import Neuron as CoreNeuron
from neurom.analysis.morphtree import i_section_radial_dist
from neurom.analysis.morphtree import trunk_section_length
from neurom.analysis.morphtree import compare_trees
import math
import numpy as np
import neurite_features as _neuf
from decorator import decorator as _dec
from neurom.core.neuron import Neuron

@_dec
def _make_iterable(func, *args, **kwargs):
	obj = args[0]
	neurons = ([obj] if isinstance(obj, Neuron)
                else (obj.neurons if hasattr(obj, 'neurons') else obj))
	args = tuple([neurons] + [a for i, a in enumerate(args) if i > 0])
	return func(*args, **kwargs)

@_make_iterable
def soma_radius(neurons):
    '''Get the radius of the soma'''
    return (nrn.soma.radius for nrn in neurons)

@_make_iterable
def soma_surface_area(neurons):
    '''Get the surface area of the soma.

    Note:
        The surface area is calculated by assuming the soma is spherical.
    '''
    return (4. * math.pi * nrn.get_soma_radius() ** 2 for nrn in neurons)


