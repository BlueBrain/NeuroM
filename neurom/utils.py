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

'''NeuroM helper utilities'''
import functools
import warnings


def memoize(fun):
    '''Memoize a function

    Caches return values based on function arguments.

    Note:
        Does not cache calls with keyword arguments.
    '''
    _cache = {}

    @functools.wraps(fun)
    def memoizer(*args, **kwargs):
        '''Return cahced value

        Calculate and store if args not in cache.
        '''
        if args not in _cache or kwargs:
            _cache[args] = fun(*args, **kwargs)
        return _cache[args]
    return memoizer


def _warn_deprecated(msg):
    '''Issue a deprecation warning'''
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
    warnings.simplefilter('default', DeprecationWarning)


def deprecated(fun_name=None, msg=""):
    '''Issue a deprecation warning for a function'''
    def _deprecated(fun):
        '''Issue a deprecation warning for a function'''
        @functools.wraps(fun)
        def _wrapper(*args, **kwargs):
            '''Issue deprecation warning and forward arguments to fun'''
            name = fun_name if fun_name is not None else fun.__name__
            _warn_deprecated('Call to deprecated function %s. %s' % (name, msg))
            return fun(*args, **kwargs)

        return _wrapper

    return _deprecated


def deprecated_module(mod_name, msg=""):
    '''Issue a deprecation warning for a module'''
    _warn_deprecated('Module %s is deprecated. %s' % (mod_name, msg))
