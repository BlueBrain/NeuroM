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

"""Type enumerations."""
import collections.abc
from enum import Enum, EnumMeta, unique

from morphio import SectionType

from neurom.utils import OrderedEnum

_SOMA_SUBTYPE = 31
_ALL_SUBTYPE = 32


@unique
class NeuriteIter(OrderedEnum):
    """Neurite iteration orders."""

    FileOrder = 1  # Order in which neurites appear in the file

    # NRN simulator order: soma -> axon -> basal -> apical
    # Coming from:
    # https://github.com/neuronsimulator/nrn/blob/2dbf2ebf95f1f8e5a9f0565272c18b1c87b2e54c/share/lib/hoc/import3d/import3d_gui.hoc#L874
    NRN = 2


def is_composite_type(subtype):
    """Check that the given type is composite."""
    return NeuriteType(subtype).is_composite()


def _is_sequence(obj):
    return isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str)


def _int_or_tuple(values):
    if isinstance(values, Enum):
        return _int_or_tuple(values.value)

    if isinstance(values, (int, SectionType)):
        return int(values)

    if _is_sequence(values):
        if len(values) == 1:
            return _int_or_tuple(values[0])
        return tuple(_int_or_tuple(v) for v in values)

    raise ValueError(f"Could not cast {values} to int or tuple of ints.")


# pylint: disable=redefined-builtin
class _ArgsIntsOrTuples(EnumMeta):
    def __call__(cls, value, names=None, *, module=None, qualname=None, type=None, start=1):
        try:
            value = _int_or_tuple(value)
        except ValueError:
            pass
        kwargs = {}
        if names is not None:
            # Keep default value of EnumMeta for Python>=3.12.3
            kwargs["names"] = names  # pragma: no cover
        return super().__call__(
            value, module=module, qualname=qualname, type=type, start=start, **kwargs
        )


def _create_neurite_type(cls, value, name=None):
    """Construct and return a cls type."""
    obj = object.__new__(cls)

    # this is an optimization to avoid checks during runtime
    if _is_sequence(value):
        subtypes = value
        root_type = value[0]
    else:
        subtypes = (value,)
        root_type = value

    setattr(obj, "_value_", value)

    if name:
        setattr(obj, "_name_", name)

    obj.subtypes = subtypes
    obj.root_type = root_type

    return obj


# for backward compatibility with 'v1' version
class NeuriteType(Enum, metaclass=_ArgsIntsOrTuples):
    """Type of neurite."""

    axon = SectionType.axon
    apical_dendrite = SectionType.apical_dendrite
    basal_dendrite = SectionType.basal_dendrite
    undefined = SectionType.undefined
    soma = SectionType.soma
    all = SectionType.all
    custom5 = SectionType.custom5
    custom6 = SectionType.custom6
    custom7 = SectionType.custom7
    custom8 = SectionType.custom8
    custom9 = SectionType.custom9
    custom10 = SectionType.custom10
    custom11 = SectionType.custom11
    custom12 = SectionType.custom12
    custom13 = SectionType.custom13
    custom14 = SectionType.custom14
    custom15 = SectionType.custom15
    custom16 = SectionType.custom16
    custom17 = SectionType.custom17
    custom18 = SectionType.custom18
    custom19 = SectionType.custom19

    axon_carrying_dendrite = SectionType.basal_dendrite, SectionType.axon

    def __new__(cls, *values):
        """Construct a NeuriteType from class definitions."""
        return _create_neurite_type(cls, value=_int_or_tuple(values))

    def __hash__(self):
        """Return the has of the type."""
        return hash(self._value_)

    def is_composite(self):
        """Return True if the type consists of more than 1 subtypes."""
        return len(self.subtypes) > 1

    def __eq__(self, other):
        """Equal operator."""
        if not isinstance(other, NeuriteType):
            try:
                other = NeuriteType(other)
            except ValueError:
                return False

        if self.is_composite():
            if other.is_composite():
                is_eq = self.subtypes == other.subtypes
            else:
                is_eq = other.root_type in self.subtypes
        else:
            if other.is_composite():
                is_eq = self.root_type in other.subtypes
            else:
                is_eq = self.root_type == other.root_type
        return is_eq


#: Collection of all neurite types
NEURITES = (NeuriteType.axon, NeuriteType.apical_dendrite, NeuriteType.basal_dendrite)

ROOT_ID = -1


def tree_type_checker(*ref):
    """Tree type checker functor.

    Args:
        *ref(NeuriteType|tuple): Either a single NeuriteType, list of them or a tuple of them.

    Returns:
        Functor that takes a tree, and returns true if that tree matches any of
        NeuriteTypes in ref

    Ex:
        >>> import neurom
        >>> from neurom.core.types import NeuriteType, tree_type_checker
        >>> from neurom.core.morphology import Section, iter_neurites
        >>> m = neurom.load_morphology("tests/data/swc/Neuron.swc")
        >>>
        >>> tree_filter = tree_type_checker(NeuriteType.axon, NeuriteType.basal_dendrite)
        >>> it = iter_neurites(m, filt=tree_filter)
        >>>
        >>> tree_filter = tree_type_checker((NeuriteType.axon, NeuriteType.basal_dendrite))
        >>> it = iter_neurites(m, filt=tree_filter)
    """
    ref = tuple(ref)
    if len(ref) == 1 and isinstance(ref[0], (list, tuple)):
        # if `ref` is passed as a tuple of types
        ref = ref[0]
    # validate that all values are of NeuriteType
    ref = [NeuriteType(t) for t in ref]
    if NeuriteType.all in ref:

        def check_tree_type(_):
            """Always returns true."""
            return True

    else:

        def check_tree_type(tree):
            """Check whether tree has the same type as ref.

            Returns:
                True if ref in the same type as tree.type or ref is NeuriteType.all
            """
            return tree.type in ref

    check_tree_type.type = ref

    return check_tree_type


def dendrite_filter(n):
    """Select only dendrites."""
    # pylint: disable=consider-using-in
    return n.type == NeuriteType.basal_dendrite or n.type == NeuriteType.apical_dendrite


def axon_filter(n):
    """Select only axons."""
    return n.type == NeuriteType.axon
