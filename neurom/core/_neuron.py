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

from collections import deque
from itertools import chain

import morphio
import numpy as np
from neurom import morphmath
from neurom.core._soma import make_soma
from neurom.core.dataformat import COLS
from neurom.core.types import NeuriteIter, NeuriteType


class Section:
    """Simple recursive tree class."""

    def __init__(self, morphio_section):
        """The section constructor."""
        self.morphio_section = morphio_section

    @property
    def id(self):
        """Returns the section ID."""
        return self.morphio_section.id

    @property
    def parent(self):
        """Returns the parent section if non root section else None."""
        if self.morphio_section.is_root:
            return None
        return Section(self.morphio_section.parent)

    @property
    def children(self):
        """Returns a list of child section."""
        return [Section(child) for child in self.morphio_section.children]

    def append_section(self, section):
        """Appends a section to the current section object.

        Args:
            section (morphio.Section|morphio.mut.Section|Section|morphio.PointLevel): a section
        """
        if isinstance(section, Section):
            return self.morphio_section.append_section(section.morphio_section)
        return self.morphio_section.append_section(section)

    def is_forking_point(self):
        """Is this section a forking point?"""
        return len(self.children) > 1

    def is_bifurcation_point(self):
        """Is tree a bifurcation point?"""
        return len(self.children) == 2

    def is_leaf(self):
        """Is tree a leaf?"""
        return len(self.children) == 0

    def is_root(self):
        """Is tree the root node?"""
        return self.parent is None

    def ipreorder(self):
        """Depth-first pre-order iteration of tree nodes."""
        children = deque((self, ))
        while children:
            cur_node = children.pop()
            children.extend(reversed(cur_node.children))
            yield cur_node

    def ipostorder(self):
        """Depth-first post-order iteration of tree nodes."""
        children = [self, ]
        seen = set()
        while children:
            cur_node = children[-1]
            if cur_node not in seen:
                seen.add(cur_node)
                children.extend(reversed(cur_node.children))
            else:
                children.pop()
                yield cur_node

    def iupstream(self):
        """Iterate from a tree node to the root nodes."""
        t = self
        while t is not None:
            yield t
            t = t.parent

    def ileaf(self):
        """Iterator to all leaves of a tree."""
        return filter(Section.is_leaf, self.ipreorder())

    def iforking_point(self, iter_mode=ipreorder):
        """Iterator to forking points.

        Args:
            iter_mode: iteration mode. Default: ipreorder.
        """
        return filter(Section.is_forking_point, iter_mode(self))

    def ibifurcation_point(self, iter_mode=ipreorder):
        """Iterator to bifurcation points.

        Args:
            iter_mode: iteration mode. Default: ipreorder.
        """
        return filter(Section.is_bifurcation_point, iter_mode(self))

    def __eq__(self, other):
        """Equal when its morphio section is equal."""
        return self.morphio_section == other.morphio_section

    def __hash__(self):
        """Hash of its id."""
        return self.id

    def __nonzero__(self):
        """If has children."""
        return self.morphio_section is not None

    __bool__ = __nonzero__

    @property
    def points(self):
        """Returns the section list of points the NeuroM way (points + radius)."""
        return np.concatenate((self.morphio_section.points,
                               self.morphio_section.diameters[:, np.newaxis] / 2.),
                              axis=1)

    @points.setter
    def points(self, value):
        """Set the points."""
        self.morphio_section.points = np.copy(value[:, COLS.XYZ])
        self.morphio_section.diameters = np.copy(value[:, COLS.R]) * 2

    @property
    def type(self):
        """Returns the section type."""
        return NeuriteType(int(self.morphio_section.type))

    @property
    def length(self):
        """Return the path length of this section."""
        return morphmath.section_length(self.points)

    @property
    def area(self):
        """Return the surface area of this section.

        The area is calculated from the segments, as defined by this
        section's points
        """
        return sum(morphmath.segment_area(s) for s in iter_segments(self))

    @property
    def volume(self):
        """Return the volume of this section.

        The volume is calculated from the segments, as defined by this
        section's points
        """
        return sum(morphmath.segment_volume(s) for s in iter_segments(self))

    def __repr__(self):
        """Text representation."""
        parent_id = None if self.parent is None else self.parent.id
        return (f'Section(id={self.id}, type={self.type}, n_points={len(self.points)})'
                f'<parent: Section(id={parent_id}), nchildren: {len(self.children)}>')


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
                  iterator_type=Section.ipreorder,
                  neurite_filter=None,
                  neurite_order=NeuriteIter.FileOrder):
    """Iterator to the sections in a neurite, neuron or neuron population.

    Arguments:
        neurites: neuron, population, neurite, or iterable containing neurite objects
        iterator_type: section iteration order within a given neurite. Must be one of:
            Section.ipreorder: Depth-first pre-order iteration of tree nodes
            Section.ipostorder: Depth-first post-order iteration of tree nodes
            Section.iupstream: Iterate from a tree node to the root nodes
            Section.ibifurcation_point: Iterator to bifurcation points
            Section.ileaf: Iterator to all leaves of a tree

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


def graft_neuron(section):
    """Returns a neuron starting at section."""
    assert isinstance(section, Section)
    m = morphio.mut.Morphology()
    m.append_root_section(section.morphio_section)
    return Neuron(m)


class Neurite:
    """Class representing a neurite tree."""

    def __init__(self, root_node):
        """Constructor.

        Args:
            root_node (morphio.Section): root section
        """
        self.morphio_root_node = root_node

    @property
    def root_node(self):
        """The first section of the neurite."""
        return Section(self.morphio_root_node)

    @property
    def type(self):
        """The type of the root node."""
        return self.root_node.type

    @property
    def points(self):
        """Array with all the points in this neurite.

        Note: Duplicate points at section bifurcations are removed
        """
        # Neurite first point must be added manually
        _ptr = [self.root_node.points[0][COLS.XYZR]]
        for s in iter_sections(self):
            _ptr.append(s.points[1:, COLS.XYZR])
        return np.vstack(_ptr)

    @property
    def length(self):
        """Returns the total length of this neurite.

        The length is defined as the sum of lengths of the sections.
        """
        return sum(s.length for s in self.iter_sections())

    @property
    def area(self):
        """Return the surface area of this neurite.

        The area is defined as the sum of area of the sections.
        """
        return sum(s.area for s in self.iter_sections())

    @property
    def volume(self):
        """Return the volume of this neurite.

        The volume is defined as the sum of volumes of the sections.
        """
        return sum(s.volume for s in self.iter_sections())

    def iter_sections(self, order=Section.ipreorder, neurite_order=NeuriteIter.FileOrder):
        """Iteration over section nodes.

        Arguments:
            order: section iteration order within a given neurite. Must be one of:
                Section.ipreorder: Depth-first pre-order iteration of tree nodes
                Section.ipreorder: Depth-first post-order iteration of tree nodes
                Section.iupstream: Iterate from a tree node to the root nodes
                Section.ibifurcation_point: Iterator to bifurcation points
                Section.ileaf: Iterator to all leaves of a tree

            neurite_order: order upon which neurites should be iterated. Values:
                - NeuriteIter.FileOrder: order of appearance in the file
                - NeuriteIter.NRN: NRN simulator order: soma -> axon -> basal -> apical
        """
        return iter_sections(self, iterator_type=order, neurite_order=neurite_order)

    def __nonzero__(self):
        """If has root node."""
        return bool(self.morphio_root_node)

    def __eq__(self, other):
        """If root node ids and types are equal."""
        return self.type == other.type and self.morphio_root_node.id == other.morphio_root_node.id

    def __hash__(self):
        """Hash is made of tuple of type and root_node."""
        return hash((self.type, self.root_node))

    __bool__ = __nonzero__

    def __repr__(self):
        """Return a string representation."""
        return 'Neurite <type: %s>' % self.type


class Neuron(morphio.mut.Morphology):
    """Class representing a simple neuron."""

    def __init__(self, filename, name=None):
        """Neuron constructor.

        Args:
            filename (str|Path): a filename
            name (str): a option neuron name
        """
        try:
            morphio.set_ignored_warning([morphio.Warning.appending_empty_section,
                                         morphio.Warning.wrong_root_point], True)
            morphio.set_raise_warnings(True)
            super().__init__(filename)
        finally:
            morphio.set_raise_warnings(False)
        self.name = name if name else 'Neuron'
        self.morphio_soma = super().soma
        self.neurom_soma = make_soma(self.morphio_soma)

    @property
    def soma(self):
        """Corresponding soma."""
        return self.neurom_soma

    @property
    def neurites(self):
        """The list of neurites."""
        return [Neurite(root_section) for root_section in self.root_sections]

    @property
    def sections(self):
        """The array of all sections, excluding the soma."""
        return list(iter_sections(self))

    @property
    def points(self):
        """Returns the list of points."""
        return np.concatenate(
            [section.points for section in iter_sections(self)])

    def transform(self, trans):
        """Return a copy of this neuron with a 3D transformation applied."""
        obj = Neuron(self)
        obj.morphio_soma.points = trans(obj.morphio_soma.points)

        for section in obj.sections:
            section.morphio_section.points = trans(section.morphio_section.points)
        return obj

    def __copy__(self):
        """Creates a deep copy of Neuron instance."""
        return Neuron(self, self.name)

    def __deepcopy__(self, memodict={}):
        """Creates a deep copy of Neuron instance."""
        # pylint: disable=dangerous-default-value
        return Neuron(self, self.name)

    def __repr__(self):
        """Return a string representation."""
        return 'Neuron <soma: %s, n_neurites: %d>' % \
            (self.soma, len(self.neurites))
