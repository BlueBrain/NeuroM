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


class SubtypeCollection(int):
    """The subtype use by the NeuriteType."""

    _BASE = 100

    def __new__(cls, *value):
        """Create an int representing a SubtypeCollection.

        Args:
            value (Union[int, Sequence[int], SubtypeCollection, NeuriteType, morphio.SectionType]):

        """
        if len(value) == 1:
            # Avoid recursion error
            value = value[0]

        if isinstance(value, collections.abc.Sequence):
            value = cls._ids_to_index([int(v) for v in value], cls._BASE)
        else:
            value = int(value)

        obj = super().__new__(cls, value)

        obj.subtypes = tuple(
            SectionType(int_type) for int_type in cls._index_to_ids(int(obj), cls._BASE)
        )

        obj._value_ = value
        return obj

    @staticmethod
    def _ids_to_index(ids, base):
        """Combine ids on a square grid with side 'base' into a single linear index."""
        if len(ids) == 1:
            return ids[0]
        return int(np.ravel_multi_index(ids, (base,) * len(ids)))

    @staticmethod
    def _index_to_ids(index, base):
        """Convert a linear index into ids on a square grid with side 'base'."""
        # find number of integers in linear index
        if index < base:
            return [index]
        ratio = math.log(index) / math.log(base)
        n_digits = math.ceil(ratio)
        if int(ratio) == n_digits:
            n_digits += 1

        int_types = np.unravel_index(index, shape=(base,) * n_digits)

        if _ALL_SUBTYPE in int_types and len(int_types) > 1:
            raise NeuroMError(
                f"A subtype containing the value {_ALL_SUBTYPE} must contain only one element "
                f"(current elements: {int_types})."
            )
        return int_types

    def __hash__(self):
        """Compute the hash of the object."""
        return hash(self._value_)

    def is_composite(self):
        """Check that the object is composite."""
        return self._value_ >= self._BASE

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
        if self.is_composite():
            if other.is_composite():
                is_eq = self.subtypes == other.subtypes
            else:
                is_eq = other._value_ in self.subtypes
        else:
            if other.is_composite():
                is_eq = self._value_ in other.subtypes
            else:
                is_eq = self._value_ == other._value_
        return is_eq

    def __ne__(self, other):
        """Not equal operator."""
        return not self == other

    @property
    def root_type(self):
        """Get the root type of a composite type."""
        return self.subtypes[0]

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

    axon = SectionType.axon
    apical_dendrite = SectionType.apical_dendrite
    basal_dendrite = SectionType.basal_dendrite
    undefined = SectionType.undefined
    soma = _SOMA_SUBTYPE
    all = _ALL_SUBTYPE
    custom5 = SectionType.custom5
    custom6 = SectionType.custom6
    custom7 = SectionType.custom7
    custom8 = SectionType.custom8
    custom9 = SectionType.custom9
    custom10 = SectionType.custom10

    # Composite types
    axon_carrying_dendrite = SectionType.basal_dendrite, SectionType.axon

    @classmethod
    def register(cls, name, value):
        """Register a new value in the Enum class."""
        new_value = SubtypeCollection(value)
        err = None
        if name in cls._member_names_:
            err = (name, cls._member_map_[name].value)
        if new_value in cls._value2member_map_:
            err = (cls._value2member_map_[new_value].name, value)
        if err is not None:
            raise ValueError(
                f"The NeuriteType '{err[0]}' is already registered with the value '{err[1]}'"
            )
        obj = super(NeuriteType, cls).__new__(cls, new_value)
        obj._name_ = name
        cls._value2member_map_[new_value] = obj
        cls._member_map_["name"] = obj
        cls._member_names_.append(name)
        return obj

    @classmethod
    def unregister(cls, name):
        """Unregister a value in the Enum class."""
        if name not in cls._member_names_:
            raise ValueError(
                f"The NeuriteType '{name}' is not registered so it can not be unregistered"
            )

        value = cls._member_map_["name"].value
        del cls._value2member_map_[value]
        del cls._member_map_["name"]
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

    # SectionType or raw integer
    elif isinstance(value, collections.abc.Hashable) and value in cls._value2member_map_:
        return cls._value2member_map_[value]

    # Composite type or unhashable type (e.g. list)
    else:
        try:
            subtype_value = SubtypeCollection(value)._value_
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        else:
            if subtype_value in cls._value2member_map_:
                return cls._value2member_map_[subtype_value]

    # Invalid value
    raise ValueError(f"{value} is not a valid registered NeuriteType")


NeuriteType.__new__ = _enum_accept_undefined


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
