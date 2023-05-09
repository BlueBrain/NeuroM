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
import math
from enum import Enum, unique

import numpy as np
from morphio import SectionType

from neurom.exceptions import NeuroMError
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


class SubtypeCollection(tuple):
    """The subtype use by the NeuriteType."""

    @staticmethod
    def _format_value(values):
        formatted_values = []
        for val in values:
            if isinstance(val, Enum):
                formatted_values.extend(SubtypeCollection._format_value(val._value_))
            elif isinstance(val, SubtypeCollection):
                formatted_values.extend(val)
            elif isinstance(val, collections.abc.Sequence):
                for i in val:
                    formatted_values.extend(SubtypeCollection._format_value([i]))
            else:
                formatted_values.append(SectionType(val))
        return formatted_values

    def __new__(cls, *value):
        if not value:
            raise ValueError("A SubtypeCollection object can not be empty")
        values = SubtypeCollection._format_value(value)
        values = tuple(values)
        if len(values) > 1 and _ALL_SUBTYPE in values:
            raise NeuroMError(
                f"A subtype containing the value {_ALL_SUBTYPE} must contain only one element "
                f"(current elements: {values})."
            )
        obj = super().__new__(cls, tuple(values))
        return obj

    def __hash__(self):
        """Compute the hash of the object."""
        return hash(tuple(self))

    # def __repr__(self):
    #     return "-".join(repr(i) for i in self.subtypes)

    def __str__(self):
        return "-".join(str(i) for i in self)

    def is_composite(self):
        """Check that the object is composite."""
        return len(self) > 1

    def __eq__(self, other):
        """Equal operator."""
        if not isinstance(other, SubtypeCollection):
            try:
                other = SubtypeCollection(other)
            except Exception:  # pylint: disable=broad-exception-caught
                # If other can not be casted to SubtypeCollection it is not equal
                return False
        # if _ALL_SUBTYPE == self._value_ or _ALL_SUBTYPE == other._value_:
        #     # This could be used to simplify the internal code of NeuroM
        #     return True

        if len(other) == 0:
            return False

        if self.is_composite():
            if other.is_composite():
                is_eq = tuple(self) == tuple(other)
            else:
                is_eq = other[0] in self
        else:
            if other.is_composite():
                is_eq = self[0] in other
            else:
                is_eq = self[0] == other[0]
        return is_eq

    def __ne__(self, other):
        """Not equal operator."""
        return not self == other

    @property
    def root_type(self):
        """Get the root type of a composite type."""
        return self[0]

    def __reduce_ex__(self, *args, **kwargs):
        """This is just to ensure the type is recognized as picklable by the Enum class."""
        return super().__reduce_ex__(*args, **kwargs)


def is_composite_type(subtype):
    """Check that the given type is composite."""
    return SubtypeCollection(subtype).is_composite()


# for backward compatibility with 'v1' version
class NeuriteType(SubtypeCollection, Enum):
    """Type of neurite."""

    # pylint: disable=no-member
    # pylint: disable=protected-access
    # pylint: disable=attribute-defined-outside-init

    axon = (SectionType.axon,)
    apical_dendrite = (SectionType.apical_dendrite,)
    basal_dendrite = (SectionType.basal_dendrite,)
    undefined = (SectionType.undefined,)
    soma = (_SOMA_SUBTYPE,)
    all = (_ALL_SUBTYPE,)
    custom5 = (SectionType.custom5,)
    custom6 = (SectionType.custom6,)
    custom7 = (SectionType.custom7,)
    custom8 = (SectionType.custom8,)
    custom9 = (SectionType.custom9,)
    custom10 = (SectionType.custom10,)

    @classmethod
    def register(cls, name, value):
        """Register a new value in the Enum class."""
        new_value = SubtypeCollection(value)
        new_value_as_tuple = tuple(new_value)
        err = None
        if name in cls._member_names_:
            err = (name, cls._member_map_[name].value)
        if new_value_as_tuple in cls._value2member_map_:
            err = (cls._value2member_map_[new_value_as_tuple].name, value)
        if err is not None:
            raise ValueError(
                f"The NeuriteType '{err[0]}' is already registered with the value '{err[1]}'"
            )
        obj = super().__new__(cls, new_value)
        obj._name_ = name
        obj._value_ = new_value
        cls._value2member_map_[new_value] = obj
        cls._member_map_[name] = obj
        cls._member_names_.append(name)
        return obj

    @classmethod
    def unregister(cls, name):
        """Unregister a value in the Enum class."""
        if name not in cls._member_names_:
            raise ValueError(
                f"The NeuriteType '{name}' is not registered so it can not be unregistered"
            )

        value = cls._member_map_[name].value
        del cls._value2member_map_[value]
        del cls._member_map_[name]
        cls._member_names_.remove(name)


def _enum_accept_undefined(cls, value):
    # pylint: disable=protected-access

    # Use NeuriteType name
    if isinstance(value, NeuriteType):
        value_str = value.name
        if value_str in cls._member_map_:
            return cls._member_map_[value_str]

    # Name given as string
    elif isinstance(value, str):
        if value in cls._member_map_:
            return cls._member_map_[value]

    # Composite type or unhashable type (e.g. list)
    else:
        try:
            subtype_value = tuple(SubtypeCollection(value))
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        else:
            if subtype_value in cls._value2member_map_:
                return cls._value2member_map_[subtype_value]

    # Invalid value
    raise ValueError(f"{value} is not a valid registered NeuriteType")


NeuriteType.__new__ = _enum_accept_undefined

# Register composite types
NeuriteType.register("axon_carrying_dendrite", (SectionType.basal_dendrite, SectionType.axon))


#: Collection of all neurite types
NEURITES = (NeuriteType.axon, NeuriteType.apical_dendrite, NeuriteType.basal_dendrite)

ROOT_ID = -1


def tree_type_checker(*ref):
    """Tree type checker functor.

    Args:
        *ref(NeuriteType|tuple): Either a single NeuriteType or a variable list of them or a tuple
        of them.

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
