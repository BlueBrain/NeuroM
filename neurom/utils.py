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

"""NeuroM helper utilities."""
import json
import warnings
from enum import Enum
from functools import wraps
from itertools import chain

import numpy as np

from neurom.core.dataformat import COLS
from neurom.exceptions import NeuroMDeprecationWarning


def warn_deprecated(msg):
    """Issue a deprecation warning."""
    warnings.warn(msg, category=NeuroMDeprecationWarning, stacklevel=3)


def deprecated(fun_name=None, msg=""):
    """Issue a deprecation warning for a function."""

    def _deprecated(fun):
        """Issue a deprecation warning for a function."""

        @wraps(fun)
        def _wrapper(*args, **kwargs):
            """Issue deprecation warning and forward arguments to fun."""
            name = fun_name if fun_name is not None else fun.__name__
            warn_deprecated('Call to deprecated function %s. %s' % (name, msg))
            return fun(*args, **kwargs)

        return _wrapper

    return _deprecated


def deprecated_module(msg):
    """Issue a deprecation warning for a module."""
    warn_deprecated(msg)


class NeuromJSON(json.JSONEncoder):
    """JSON encoder that handles numpy types.

    In python3, numpy.dtypes don't serialize to correctly, so a custom
    converter is needed.
    """

    def default(self, o):  # pylint: disable=method-hidden
        """Override default method for numpy types."""
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


# pylint: disable=comparison-with-callable
class OrderedEnum(Enum):
    """An ordered enum class.

    Implementation taken here https://docs.python.org/3/library/enum.html#orderedenum

    Fixes https://github.com/BlueBrain/NeuroM/issues/697
    """

    def __ge__(self, other):
        """Check greater than or equal to."""
        if self.__class__ is other.__class__:
            return self.value >= other.value
        raise NotImplementedError

    def __gt__(self, other):
        """Check greater than."""
        if self.__class__ is other.__class__:
            return self.value > other.value
        raise NotImplementedError

    def __le__(self, other):
        """Check less than or equal to."""
        if self.__class__ is other.__class__:
            return self.value <= other.value
        raise NotImplementedError

    def __lt__(self, other):
        """Check less than."""
        if self.__class__ is other.__class__:
            return self.value < other.value
        raise NotImplementedError


def str_to_plane(plane):
    """Transform a plane string into a list of coordinates."""
    if plane is not None:
        coords = []
        plane = plane.lower()
        if "x" in plane:
            coords.append(COLS.X)
        if "y" in plane:
            coords.append(COLS.Y)
        if "z" in plane:
            coords.append(COLS.Z)
    else:  # pragma: no cover
        coords = COLS.XYZ
    return coords


def flatten(list_of_lists):
    """Flatten one level of nesting."""
    return chain.from_iterable(list_of_lists)


def filtered_iterator(predicate, iterator_type):
    """Returns an iterator function that is filtered by the predicate."""

    @wraps(iterator_type)
    def composed(*args, **kwargs):
        return filter(predicate, iterator_type(*args, **kwargs))

    return composed
