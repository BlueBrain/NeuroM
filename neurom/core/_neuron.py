# Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
# All rights reserved.
#
# This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the names of
#        its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Neuron classes and functions."""

from copy import deepcopy
from itertools import chain

import numpy as np

from neurom import morphmath
from neurom.core._soma import Soma
from neurom.core.dataformat import COLS
from neurom.utils import memoize

from . import NeuriteType, Tree, NeuriteIter

# NRN simulator iteration order
# See:
# https://github.com/neuronsimulator/nrn/blob/2dbf2ebf95f1f8e5a9f0565272c18b1c87b2e54c/share/lib/hoc/import3d/import3d_gui.hoc#L874
NRN_ORDER = {NeuriteType.soma: 0,
             NeuriteType.axon: 1,
             NeuriteType.basal_dendrite: 2,
             NeuriteType.apical_dendrite: 3,
             NeuriteType.undefined: 4}


def iter_neurites(obj, mapfun=None, filt=None, neurite_order=NeuriteIter.FileOrder):
    """Iterator to a neurite, neuron or neuron population.

    Applies optional neurite filter and mapping functions.

    Arguments:
        obj: a neurite, neuron or neuron population.
        mapfun: optional neurite mapping function.
        filt: optional neurite filter function.
        neurite_order (NeuriteIter): order upon which neurites should be iterated
            - NeuriteIter.FileOrder: order of appearance in the file
            - NeuriteIter.NRN: NRN simulator order: soma -> axon -> basal -> apical

    Examples:
        Get the number of points in each neurite in a neuron population

        >>> from neurom.core import iter_neurites
        >>> n_points = [n for n in iter_neurites(pop, lambda x : len(x.points))]

        Get the number of points in each axon in a neuron population

        >>> import neurom as nm
        >>> from neurom.core import iter_neurites
        >>> filter = lambda n : n.type == nm.AXON
        >>> mapping = lambda n : len(n.points)
        >>> n_points = [n for n in iter_neurites(pop, mapping, filter)]
    """
    neurites = ((obj,) if isinstance(obj, Neurite) else
                obj.neurites if hasattr(obj, 'neurites') else obj)
    if neurite_order == NeuriteIter.NRN:
        last_position = max(NRN_ORDER.values()) + 1
        neurites = sorted(neurites, key=lambda neurite: NRN_ORDER.get(neurite.type, last_position))

    neurite_iter = iter(neurites) if filt is None else filter(filt, neurites)
    return neurite_iter if mapfun is None else map(mapfun, neurite_iter)


def iter_sections(neurites,
                  iterator_type=Tree.ipreorder,
                  neurite_filter=None,
                  neurite_order=NeuriteIter.FileOrder):
    """Iterator to the sections in a neurite, neuron or neuron population.

    Arguments:
        neurites: neuron, population, neurite, or iterable containing neurite objects
        iterator_type: section iteration order within a given neurite. Must be one of:
            Tree.ipreorder: Depth-first pre-order iteration of tree nodes
            Tree.ipostorder: Depth-first post-order iteration of tree nodes
            Tree.iupstream: Iterate from a tree node to the root nodes
            Tree.ibifurcation_point: Iterator to bifurcation points
            Tree.ileaf: Iterator to all leaves of a tree

        neurite_filter: optional top level filter on properties of neurite neurite objects.
        neurite_order (NeuriteIter): order upon which neurites should be iterated
            - NeuriteIter.FileOrder: order of appearance in the file
            - NeuriteIter.NRN: NRN simulator order: soma -> axon -> basal -> apical


    Examples:
        Get the number of points in each section of all the axons in a neuron population

        >>> import neurom as nm
        >>> from neurom.core import ites_sections
        >>> filter = lambda n : n.type == nm.AXON
        >>> n_points = [len(s.points) for s in iter_sections(pop,  neurite_filter=filter)]
    """
    return chain.from_iterable(
        iterator_type(neurite.root_node) for neurite in
        iter_neurites(neurites, filt=neurite_filter, neurite_order=neurite_order))


def iter_segments(obj, neurite_filter=None, neurite_order=NeuriteIter.FileOrder):
    """Return an iterator to the segments in a collection of neurites.

    Arguments:
        obj: neuron, population, neurite, section, or iterable containing neurite objects
        neurite_filter: optional top level filter on properties of neurite neurite objects
        neurite_order: order upon which neurite should be iterated. Values:
            - NeuriteIter.FileOrder: order of appearance in the file
            - NeuriteIter.NRN: NRN simulator order: soma -> axon -> basal -> apical

    Note:
        This is a convenience function provided for generic access to
        neuron segments. It may have a performance overhead WRT custom-made
        segment analysis functions that leverage numpy and section-wise iteration.
    """
    sections = iter((obj,) if isinstance(obj, Section) else
                    iter_sections(obj,
                                  neurite_filter=neurite_filter,
                                  neurite_order=neurite_order))

    return chain.from_iterable(zip(sec.points[:-1], sec.points[1:])
                               for sec in sections)


def graft_neuron(root_section):
    """Returns a neuron starting at root_section."""
    assert isinstance(root_section, Section)
    return Neuron(soma=Soma(root_section.points[:1]), neurites=[Neurite(root_section)])


class Section(Tree):
    """Class representing a neurite section."""

    def __init__(self, points, section_id=None, section_type=NeuriteType.undefined):
        """Initialize a Section object."""
        super().__init__()
        self.id = section_id
        self.points = points
        self.type = section_type

    @property
    @memoize
    def length(self):
        """Return the path length of this section."""
        return morphmath.section_length(self.points)

    @property
    @memoize
    def area(self):
        """Return the surface area of this section.

        The area is calculated from the segments, as defined by this
        section's points
        """
        return sum(morphmath.segment_area(s) for s in iter_segments(self))

    @property
    @memoize
    def volume(self):
        """Return the volume of this section.

        The volume is calculated from the segments, as defined by this
        section's points
        """
        return sum(morphmath.segment_volume(s) for s in iter_segments(self))

    def __str__(self):
        """Return a string representation."""
        return 'Section(id=%s, type=%s, n_points=%s) <parent: %s, nchildren: %d>' % \
            (self.id, self.type, len(self.points), self.parent, len(self.children))

    __repr__ = __str__


class Neurite(object):
    """Class representing a neurite tree."""

    def __init__(self, root_node):
        """Initialize a Neurite object."""
        self.root_node = root_node
        self.type = root_node.type if hasattr(
            root_node, 'type') else NeuriteType.undefined

    @property
    @memoize
    def points(self):
        """Return unordered array with all the points in this neurite."""
        # add all points in a section except the first one, which is a duplicate
        _pts = [v for s in self.root_node.ipreorder()
                for v in s.points[1:, COLS.XYZR]]
        # except for the very first point, which is not a duplicate
        _pts.insert(0, self.root_node.points[0][COLS.XYZR])
        return np.array(_pts)

    @property
    @memoize
    def length(self):
        """Return the total length of this neurite.

        The length is defined as the sum of lengths of the sections.
        """
        return sum(s.length for s in self.iter_sections())

    @property
    @memoize
    def area(self):
        """Return the surface area of this neurite.

        The area is defined as the sum of area of the sections.
        """
        return sum(s.area for s in self.iter_sections())

    @property
    @memoize
    def volume(self):
        """Return the volume of this neurite.

        The volume is defined as the sum of volumes of the sections.
        """
        return sum(s.volume for s in self.iter_sections())

    def transform(self, trans):
        """Return a copy of this neurite with a 3D transformation applied."""
        clone = deepcopy(self)
        for n in clone.iter_sections():
            n.points[:, 0:3] = trans(n.points[:, 0:3])

        return clone

    def iter_sections(self, order=Tree.ipreorder, neurite_order=NeuriteIter.FileOrder):
        """Iteration over section nodes.

        Arguments:
            order: section iteration order within a given neurite. Must be one of:
                Tree.ipreorder: Depth-first pre-order iteration of tree nodes
                Tree.ipreorder: Depth-first post-order iteration of tree nodes
                Tree.iupstream: Iterate from a tree node to the root nodes
                Tree.ibifurcation_point: Iterator to bifurcation points
                Tree.ileaf: Iterator to all leaves of a tree

            neurite_order: order upon which neurites should be iterated. Values:
                - NeuriteIter.FileOrder: order of appearance in the file
                - NeuriteIter.NRN: NRN simulator order: soma -> axon -> basal -> apical
        """
        return iter_sections(self, iterator_type=order, neurite_order=neurite_order)

    def __deepcopy__(self, memo):
        """Deep copy of neurite object."""
        return Neurite(deepcopy(self.root_node, memo))

    def __nonzero__(self):
        """Check non-zero."""
        return bool(self.root_node)

    def __eq__(self, other):
        """Check equality."""
        return self.type == other.type and self.root_node == other.root_node

    def __hash__(self):
        """Return object hash."""
        return hash((self.type, self.root_node))

    __bool__ = __nonzero__

    def __str__(self):
        """Return a string representation."""
        return 'Neurite <type: %s>' % self.type

    __repr__ = __str__


class Neuron(object):
    """Class representing a simple neuron."""

    def __init__(self, soma=None, neurites=None, sections=None, name='Neuron'):
        """Initialize a Neuron object."""
        self.soma = soma
        self.name = name
        self.neurites = neurites
        self.sections = sections

    def __str__(self):
        """Return a string representation."""
        return 'Neuron <soma: %s, n_neurites: %d>' % \
            (self.soma, len(self.neurites))

    __repr__ = __str__
