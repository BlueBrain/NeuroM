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
from enum import Enum, unique

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
        """Create new objects."""
        if len(value) == 1:
            # Avoid recursion error
            value = value[0]

        if isinstance(value, (list, tuple)):
            value = cls.from_list(value)
        elif isinstance(value, Enum):
            value = value.value

        obj = super().__new__(cls, value)

        if isinstance(value, SubtypeCollection):
            obj._subtypes = list(value._subtypes)
        else:
            obj._subtypes = SubtypeCollection.to_list(obj)
        obj._value_ = int(obj)
        return obj

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
                is_eq = self._subtypes == other._subtypes
            else:
                is_eq = other._value_ in self._subtypes
        else:
            if other.is_composite():
                is_eq = self._value_ in other._subtypes
            else:
                is_eq = self._value_ == other._value_
        return is_eq

    def __ne__(self, other):
        """Not equal operator."""
        return not self == other

    @classmethod
    def from_list(cls, values):
        """Create a NeuriteType from a list of sub types."""
        value = 0
        for num, i in enumerate(values[::-1]):
            new_value = cls._BASE**num * int(i)
            value += new_value
        obj = cls(value)
        return obj

    def to_list(self):
        """Transform a NeuriteType or a positive integer in a list of sub types."""
        digits = []
        tmp_value = int(self)
        while tmp_value:
            digits.append(int(tmp_value % self._BASE))
            tmp_value //= self._BASE
        if not digits:
            digits = [0]
        res = digits[::-1]
        if _ALL_SUBTYPE in res and len(res) > 1:
            raise NeuroMError(
                f"A subtype containing the value {_ALL_SUBTYPE} must contain only one element "
                f"(current elements: {res})."
            )
        return res

    @property
    def root_type(self):
        """Get the root type of a composite type."""
        return self.to_list()[0]

    def __reduce_ex__(self, *args, **kwargs):
        """This is just to ensure the type is recognized as picklable by the Enum class."""
        return super().__reduce_ex__(*args, **kwargs)


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
    try:
        obj = cls._member_map_[value]
    except (KeyError, TypeError):
        try:
            subtype_value = SubtypeCollection(value)
            obj = cls._value2member_map_[subtype_value]
        except (KeyError, ValueError) as exc2:
            raise ValueError(f"{value} is not a valid NeuriteType") from exc2
    return obj


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
    if len(ref) == 1 and isinstance(ref[0], tuple):
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
