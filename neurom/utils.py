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
from functools import partial, update_wrapper, wraps

import numpy as np


class memoize:
    """cache the return value of a method.

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method::

       class Obj:
           @memoize
           def add_to(self, arg):
               return self + arg

       Obj.add_to(1) # not enough arguments
       Obj.add_to(1, 2) # returns 3, result is not cached
    """

    def __init__(self, func):
        """Initialize a memoize object."""
        self.func = func
        update_wrapper(self, func)

    def __get__(self, obj, objtype=None):
        """Get the attribute from the object."""
        return partial(self, obj)

    def __call__(self, *args, **kw):
        """Callable for decorator."""
        obj = args[0]
        try:
            cache = obj.__cache  # pylint: disable=protected-access
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


def _warn_deprecated(msg):
    """Issue a deprecation warning."""
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
    warnings.simplefilter('default', DeprecationWarning)


def deprecated(fun_name=None, msg=""):
    """Issue a deprecation warning for a function."""
    def _deprecated(fun):
        """Issue a deprecation warning for a function."""
        @wraps(fun)
        def _wrapper(*args, **kwargs):
            """Issue deprecation warning and forward arguments to fun."""
            name = fun_name if fun_name is not None else fun.__name__
            _warn_deprecated('Call to deprecated function %s. %s' % (name, msg))
            return fun(*args, **kwargs)

        return _wrapper

    return _deprecated


def deprecated_module(mod_name, msg=""):
    """Issue a deprecation warning for a module."""
    _warn_deprecated('Module %s is deprecated. %s' % (mod_name, msg))


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
