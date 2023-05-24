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


class SubtypeCollection:
    """The subtype use by the NeuriteType."""

    @staticmethod
    def _format_value(values):
        # pylint: disable=protected-access
        formatted_values = []
        for val in values:
            if isinstance(val, Enum):
                formatted_values.extend(SubtypeCollection._format_value([val._value_]))
            elif isinstance(val, SubtypeCollection):
                formatted_values.extend(val.subtypes)
            elif isinstance(val, collections.abc.Sequence) and not isinstance(val, str):
                for i in val:
                    formatted_values.extend(SubtypeCollection._format_value([i]))
            else:
                formatted_values.append(val)

        return tuple(SectionType(s) for s in formatted_values)

    def __init__(self, *value):
        """Create an tuple representing a SubtypeCollection.

        Args:
            value (Union[
                int, SubtypeCollection, NeuriteType, morphio.SectionType,
                Sequence[int],
                Sequence[SubtypeCollection],
                Sequence[NeuriteType],
                Sequence[morphio.SectionType]
                ]): The value(s) of the subtype.
        """
        if not value:
            raise ValueError("A SubtypeCollection object can not be empty")
        values = self._format_value(value)
        if len(values) > 1 and _ALL_SUBTYPE in values:
            raise NeuroMError(
                f"A subtype containing the value {_ALL_SUBTYPE} must contain only one element "
                f"(current elements: {values})."
            )
        self.subtypes = values

    def __hash__(self):
        """Compute the hash of the object."""
        return hash(self.subtypes)

    def __len__(self):
        """Return number of subtypes."""
        return len(self.subtypes)

    def __repr__(self):
        """Printable representation of the object."""
        return str(self.subtypes)

    def __str__(self):
        """String representation of the object."""
        return "-".join(str(i) for i in self.subtypes)

    def __int__(self):
        """Return the integer representation of the object.

        Note:
            The subtypes are converted to a base 100 in order to be able to represent all relevant
            values.
        """
        if len(self) == 1:
            return int(self.subtypes[0])
        return int(np.ravel_multi_index([int(i) for i in self.subtypes], (100,) * len(self)))

    def is_composite(self):
        """Check that the object is composite."""
        return len(self.subtypes) > 1

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
                is_eq = other.root_type in self.subtypes
        else:
            if other.is_composite():
                is_eq = self.root_type in other.subtypes
            else:
                is_eq = self.root_type == other.root_type
        return is_eq

    def __ne__(self, other):
        """Not equal operator."""
        return not self == other

    @property
    def root_type(self):
        """Get the root type of a composite type."""
        return self.subtypes[0]


def is_composite_type(subtype):
    """Check that the given type is composite."""
    return SubtypeCollection(subtype).is_composite()


def __enum_accept_undefined__(cls, *args):
    """Replace default __new__ to accept both keys and values."""
    # pylint: disable=protected-access
    if not args:
        # Used by Pickle
        return object.__new__(cls)

    if len(args) > 1:
        raise ValueError(f"The {cls.__name__} class constructor accepts only 1 argument.")

    value = args[0]

    # Use NeuriteType name
    if isinstance(value, cls):
        value_str = value.name
        if hasattr(cls, value_str):
            return getattr(cls, value_str)

    # Name given as string
    elif isinstance(value, str):
        if hasattr(cls, value):
            return getattr(cls, value)

    # Composite type or unhashable type (e.g. list)
    else:
        try:
            subtype_value = SubtypeCollection(value).subtypes
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        else:
            if subtype_value in cls._reverse_mapping:
                return getattr(cls, cls._reverse_mapping[subtype_value])

    # Invalid value
    raise ValueError(f"{value} is not a valid registered NeuriteType")


def _empty_init(self, *args, **kwargs):  # pylint: disable=unused-argument
    """Replace default __init__ to do nothing more than the __new__ method."""


class MutableEnumType(type):
    """Metaclass to behave like a simple Enum class with mutable members."""

    # pylint: disable=attribute-defined-outside-init

    def __new__(mcs, cls_name, bases, classdict, **kwargs):
        """The constroctor of the class to update its members."""
        # Build the new class
        cls = super().__new__(mcs, cls_name, bases, classdict, **kwargs)

        # Format the attributes
        cls._reverse_mapping = {}
        for key, value in vars(cls).items():
            if key.startswith("_") or callable(value) or isinstance(value, (classmethod, property)):
                continue
            obj = cls(value)
            obj._name_ = key
            setattr(cls, key, obj)
            cls._reverse_mapping[obj.subtypes] = key

        # Set specific __new__ method to be able to get instances either by name or value
        cls.__new__ = __enum_accept_undefined__

        # Set specific __init__ method to be able to call `cls(value)` but only relying on __new__
        cls.__init__ = _empty_init

        return cls

    def __iter__(cls):
        """Iterate over registered values of the class."""
        # pylint: disable=protected-access
        return (getattr(cls, i) for i in cls._reverse_mapping.values())

    def register(cls, name, value):
        """Register a new key/value pair as attribute in the class."""
        # pylint: disable=protected-access
        obj = SubtypeCollection(value)
        if hasattr(cls, name) or obj.subtypes in cls._reverse_mapping:
            raise ValueError(f"The NeuriteType '{name}' is already registered.")
        obj.__class__ = cls
        obj._name_ = name
        setattr(cls, name, obj)
        cls._reverse_mapping[obj.subtypes] = name
        return obj

    def unregister(cls, name):
        """Unregister an attribute from the class."""
        if not hasattr(cls, name) or not isinstance(getattr(cls, name), cls):
            raise ValueError(
                f"The NeuriteType '{name}' is not registered so it can not be unregistered"
            )
        del cls._reverse_mapping[getattr(cls, name).subtypes]
        delattr(cls, name)


# for backward compatibility with 'v1' version
class NeuriteType(SubtypeCollection, metaclass=MutableEnumType):
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

    axon_carrying_dendrite = SectionType.basal_dendrite, SectionType.axon

    @property
    def name(self):
        """Get the associated name of the NeuriteType member."""
        return self._name_

    def __repr__(self):
        """Printable representation of the NeuriteType member."""
        return f"<NeuriteType.{self._name_}: {repr(self.subtypes)}>"

    def __str__(self):
        """String representation of the NeuriteType member."""
        return f"NeuriteType.{self._name_}"


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
