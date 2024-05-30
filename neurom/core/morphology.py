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

"""Morphology classes and functions."""

import warnings
from collections import deque

import morphio
import numpy as np
from cached_property import cached_property

from neurom import morphmath
from neurom.core.dataformat import COLS
from neurom.core.population import Population
from neurom.core.soma import make_soma
from neurom.core.types import NeuriteIter, NeuriteType
from neurom.exceptions import NeuroMError
from neurom.utils import flatten


class Section:
    """Simple recursive tree class."""

    def __init__(self, morphio_section):
        """The section constructor."""
        self._morphio_section = morphio_section

    def to_morphio(self):
        """Returns the morphio section."""
        return self._morphio_section

    @property
    def id(self):
        """Returns the section ID."""
        return self._morphio_section.id

    @property
    def parent(self):
        """Returns the parent section if non root section else None."""
        return None if self.is_root() else Section(self._morphio_section.parent)

    @property
    def children(self):
        """Returns a list of child section."""
        return [Section(child) for child in self._morphio_section.children]

    def is_homogeneous_point(self):
        """A section is homogeneous if it has the same type with its children."""
        return all(c.type == self.type for c in self.children)

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
        return self._morphio_section.is_root

    def ipreorder(self):
        """Depth-first pre-order iteration of tree nodes."""
        children = deque((self,))
        while children:
            cur_node = children.pop()
            children.extend(reversed(cur_node.children))
            yield cur_node

    def ipostorder(self):
        """Depth-first post-order iteration of tree nodes."""
        children = [
            self,
        ]
        seen = set()
        while children:
            cur_node = children[-1]
            if cur_node not in seen:
                seen.add(cur_node)
                children.extend(reversed(cur_node.children))
            else:
                children.pop()
                yield cur_node

    def iupstream(self, stop_node=None):
        """Iterate from a tree node to the root nodes.

        Args:
            stop_node: Node to stop the upstream traversal. If None, it stops when parent is None.
        """
        if stop_node is None:

            def stop_condition(section):
                return section.is_root()

        else:

            def stop_condition(section):
                return section.is_root() or section == stop_node

        current_section = self
        while not stop_condition(current_section):
            yield current_section
            current_section = current_section.parent
        yield current_section

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
        return self.to_morphio().has_same_shape(other.to_morphio())

    def __hash__(self):
        """Hash of its id."""
        return self.id

    @property
    def segments(self):
        """The array of all segments of the neurite."""
        return list(iter_segments(self))

    @property
    def points(self):
        """Returns the section list of points the NeuroM way (points + radius)."""
        return np.concatenate(
            (self._morphio_section.points, self._morphio_section.diameters[:, np.newaxis] / 2.0),
            axis=1,
        )

    @property
    def type(self):
        """Returns the section type."""
        return NeuriteType(int(self._morphio_section.type))

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
        return (
            f'Section(id={self.id}, type={self.type}, n_points={len(self.points)})'
            f'<parent: Section(id={parent_id}), nchildren: {len(self.children)}>'
        )


# NRN simulator iteration order
# See:
# https://github.com/neuronsimulator/nrn/blob/2dbf2ebf95f1f8e5a9f0565272c18b1c87b2e54c/share/lib/hoc/import3d/import3d_gui.hoc#L874
NRN_ORDER = {
    NeuriteType.soma: 0,
    NeuriteType.axon: 1,
    NeuriteType.basal_dendrite: 2,
    NeuriteType.apical_dendrite: 3,
    NeuriteType.undefined: 4,
}


def iter_neurites(obj, mapfun=None, filt=None, neurite_order=NeuriteIter.FileOrder):
    """Iterator to a neurite, morphology or morphology population.

    Applies optional neurite filter and mapping functions.

    Arguments:
        obj: a neurite, morphology or morphology population.
        mapfun: optional neurite mapping function.
        filt: optional neurite filter function.
        neurite_order (NeuriteIter): order upon which neurites should be iterated
            - NeuriteIter.FileOrder: order of appearance in the file
            - NeuriteIter.NRN: NRN simulator order: soma -> axon -> basal -> apical

    Examples:
        Get the number of points in each neurite in a morphology population

        >>> from neurom.core.morphology import iter_neurites
        >>> from neurom import load_morphologies
        >>> pop = load_morphologies("tests/data/valid_set")
        >>> n_points = [n for n in iter_neurites(pop, lambda x, section_type: len(x.points))]

        Get the number of points in each axon in a morphology population

        >>> import neurom as nm
        >>> from neurom.core.morphology import iter_neurites
        >>> from neurom import load_morphologies
        >>> pop = load_morphologies("tests/data/valid_set")
        >>> filter = lambda n : n.type == nm.AXON
        >>> mapping = lambda n, section_type: len(n.points)
        >>> n_points = [n for n in iter_neurites(pop, mapping, filter)]
    """
    if isinstance(obj, Neurite):
        neurites = (obj,)
    elif hasattr(obj, "neurites"):
        neurites = obj.neurites
    else:
        neurites = obj

    if neurite_order == NeuriteIter.NRN:
        if isinstance(obj, Population):
            warnings.warn(
                '`iter_neurites` with `neurite_order` over Population orders neurites'
                'within the whole population, not within each morphology separately.'
            )
        last_position = max(NRN_ORDER.values()) + 1
        neurites = sorted(neurites, key=lambda neurite: NRN_ORDER.get(neurite.type, last_position))

    neurite_iter = iter(neurites) if filt is None else filter(filt, neurites)

    if mapfun is None:
        return neurite_iter

    return (
        (
            mapfun(
                neurite,
                section_type=filt.type if filt is not None else None,
            )
            if neurite.process_subtrees
            else mapfun(neurite, section_type=NeuriteType.all)
        )
        for neurite in neurite_iter
    )


def iter_sections(
    neurites,
    iterator_type=Section.ipreorder,
    neurite_filter=None,
    neurite_order=NeuriteIter.FileOrder,
    section_filter=None,
):
    """Iterator to the sections in a neurite, morphology or morphology population.

    Arguments:
        neurites: morphology, population, neurite, or iterable containing neurite objects
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
        section_filter: optional section level filter. Please note that neurite_filter takes
            precedence over the section_filter.


    Examples:
        Get the number of points in each section of all the axons in a morphology population

        >>> import neurom as nm
        >>> from neurom.core.morphology import iter_sections
        >>> filter = lambda n : n.type == nm.AXON
        >>> n_points = [len(s.points) for s in iter_sections(pop,  neurite_filter=filter)]
    """
    neurites = iter_neurites(neurites, filt=neurite_filter, neurite_order=neurite_order)
    sections = flatten(iterator_type(neurite.root_node) for neurite in neurites)
    return sections if section_filter is None else filter(section_filter, sections)


def iter_segments(
    obj,
    neurite_filter=None,
    neurite_order=NeuriteIter.FileOrder,
    section_filter=None,
    section_iterator=Section.ipreorder,
):
    """Return an iterator to the segments in a collection of neurites.

    Arguments:
        obj: morphology, population, neurite, section, or iterable containing neurite objects
        neurite_filter: optional top level filter on properties of neurite neurite objects
        neurite_order: order upon which neurite should be iterated. Values:
            - NeuriteIter.FileOrder: order of appearance in the file
            - NeuriteIter.NRN: NRN simulator order: soma -> axon -> basal -> apical
        section_filter: optional section level filter
        section_iterator: section iteration order within a given neurite. Must be one of:
            Section.ipreorder: Depth-first pre-order iteration of tree nodes
            Section.ipostorder: Depth-first post-order iteration of tree nodes
            Section.iupstream: Iterate from a tree node to the root nodes
            Section.ibifurcation_point: Iterator to bifurcation points
            Section.ileaf: Iterator to all leaves of a tree

    Note:
        This is a convenience function provided for generic access to
        morphology segments. It may have a performance overhead WRT custom-made
        segment analysis functions that leverage numpy and section-wise iteration.
    """
    sections = iter(
        (obj,)
        if isinstance(obj, Section)
        else iter_sections(
            obj,
            iterator_type=section_iterator,
            neurite_filter=neurite_filter,
            neurite_order=neurite_order,
            section_filter=section_filter,
        )
    )

    return flatten(zip(section.points[:-1], section.points[1:]) for section in sections)


def iter_points(obj, neurite_filter=None, neurite_order=NeuriteIter.FileOrder, section_filter=None):
    """Return an iterator to the points in a population, morphology, neurites, or section.

    Args:
        obj: population, morphology, neurite, section or iterable containing
        neurite_filter: optional top level filter on properties of neurite neurite objects
        neurite_order: order upon which neurite should be iterated. Values:
            - NeuriteIter.FileOrder: order of appearance in the file
            - NeuriteIter.NRN: NRN simulator order: soma -> axon -> basal -> apical
        section_filter: optional section level filter
    """
    sections = (
        iter((obj,))
        if isinstance(obj, Section)
        else iter_sections(
            obj,
            neurite_filter=neurite_filter,
            neurite_order=neurite_order,
            section_filter=section_filter,
        )
    )

    return flatten(s.points[:, COLS.XYZ] for s in sections)


def graft_morphology(section):
    """Returns a morphology starting at section."""
    assert isinstance(section, Section)
    m = morphio.mut.Morphology()
    m.append_root_section(section.to_morphio())
    return Morphology(m)


class Neurite:
    """Class representing a neurite tree."""

    def __init__(self, root_node, *, process_subtrees=False):
        """Constructor.

        Args:
            root_node (morphio.Section): root section
            process_subtrees (bool): enable mixed tree processing if set to True
        """
        self._root_node = root_node
        self._process_subtrees = process_subtrees

    @property
    def process_subtrees(self):
        """Enable mixed tree processing if set to True."""
        return self._process_subtrees

    @process_subtrees.setter
    def process_subtrees(self, value):
        self._process_subtrees = value
        if "type" in vars(self):
            del vars(self)["type"]

    @property
    def morphio_root_node(self):
        """Returns the morphio root section."""
        return self._root_node

    @property
    def root_node(self):
        """The first section of the neurite."""
        return Section(self.morphio_root_node)

    @cached_property
    def type(self):
        """The type of the Neurite (which can be composite)."""
        return NeuriteType(self.subtree_types)

    @cached_property
    def subtree_types(self):
        """The types of the subtrees."""
        if not self._process_subtrees:
            return NeuriteType(self.morphio_root_node.type)

        it = self.root_node.ipreorder()
        subtree_types = [next(it).to_morphio().type]

        for section in it:
            # A subtree can start from a single branch or as a fork of the same type from a parent
            # with different type.
            if section.type not in (section.parent.type, subtree_types[-1]):
                subtree_types.append(section.to_morphio().type)

        return subtree_types

    @property
    def sections(self):
        """The array of all sections."""
        return list(iter_sections(self))

    @property
    def segments(self):
        """The array of all segments of the neurite."""
        return list(iter_segments(self))

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
        # pylint: disable=import-outside-toplevel
        from neurom.features.neurite import total_length

        return total_length(self)

    @property
    def area(self):
        """Return the surface area of this neurite.

        The area is defined as the sum of area of the sections.
        """
        # pylint: disable=import-outside-toplevel
        from neurom.features.neurite import total_area

        return total_area(self)

    @property
    def volume(self):
        """Return the volume of this neurite.

        The volume is defined as the sum of volumes of the sections.
        """
        # pylint: disable=import-outside-toplevel
        from neurom.features.neurite import total_volume

        return total_volume(self)

    def is_heterogeneous(self) -> bool:
        """Returns true if the neurite consists of more that one section types."""
        return self.morphio_root_node.is_heterogeneous()

    def __eq__(self, other):
        """If root node ids and types are equal."""
        return (
            self.type == other.type
            and self.morphio_root_node.id == other.morphio_root_node.id
            and self.process_subtrees == other.process_subtrees
        )

    def __hash__(self):
        """Hash is made of tuple of type and root_node."""
        return hash((self.type, self.root_node, self.process_subtrees))

    def __repr__(self):
        """Return a string representation."""
        return 'Neurite <type: %s>' % self.type


class Morphology:
    """Class representing a simple morphology."""

    def __init__(self, morphio_morph, name=None, *, process_subtrees=False):
        """Morphology constructor.

        Args:
            morphio_morph (morphio.Morphology|morphio.mut.Morphology): a morphio object
            name (str): an optional morphology name
            process_subtrees (bool): enable mixed tree processing if set to True
        """
        if not isinstance(morphio_morph, (morphio.Morphology, morphio.mut.Morphology)):
            raise NeuroMError(
                f"Expected morphio Morphology object but got: {morphio_morph}.\n"
                f"Use neurom.load_morphology() to load from file."
            )

        self._morphio_morph = morphio_morph

        self.name = name if name else 'Morphology'
        self.soma = make_soma(self._morphio_morph.soma)

        self.process_subtrees = process_subtrees

    def to_morphio(self):
        """Returns the morphio morphology object."""
        return self._morphio_morph

    def copy(self):
        """Returns a shallow copy of the morphio morphology object."""
        return Morphology(self.to_morphio(), name=self.name, process_subtrees=self.process_subtrees)

    @property
    def neurites(self):
        """The list of neurites."""
        return [
            Neurite(root_section, process_subtrees=self.process_subtrees)
            for root_section in self._morphio_morph.root_sections
        ]

    def section(self, section_id):
        """Returns the section with the given id."""
        return Section(self._morphio_morph.section(section_id))

    @property
    def sections(self):
        """The array of all sections, excluding the soma."""
        return list(iter_sections(self))

    @property
    def segments(self):
        """The array of all segments of the sections."""
        return list(iter_segments(self))

    @property
    def points(self):
        """Returns the list of points."""
        return np.concatenate([section.points for section in iter_sections(self)])

    def transform(self, trans):
        """Return a copy of this morphology with a 3D transformation applied."""
        morph = self._morphio_morph

        is_immutable = hasattr(morph, 'as_mutable')

        # make copy or convert to mutable if immutable
        if is_immutable:
            morph = morph.as_mutable()
        else:
            morph = morphio.mut.Morphology(morph)

        morph.soma.points = trans(morph.soma.points)

        for section in morph.iter():
            section.points = trans(section.points)

        if is_immutable:
            return Morphology(morph.as_immutable())
        return Morphology(morph)

    def __copy__(self):
        """Creates a deep copy of Morphology instance."""
        return Morphology(self.to_morphio(), self.name)

    def __deepcopy__(self, memodict={}):
        """Creates a deep copy of Morphology instance."""
        # pylint: disable=dangerous-default-value
        return Morphology(self.to_morphio(), self.name)

    def __repr__(self):
        """Return a string representation."""
        return 'Morphology <soma: %s, n_neurites: %d>' % (self.soma, len(self.neurites))
