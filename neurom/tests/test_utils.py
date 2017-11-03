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

'''Test neurom.utils'''
import json
import numpy as np

from neurom import utils as nu
from nose import tools as nt
import random
import warnings


def test_memoize_caches_args():

    @nu.memoize
    def dummy(x, y=42):
        return random.random()

    ref1 = dummy(42)
    ref2 = dummy(42, 43)

    for _ in range(10):
        nt.assert_equal(dummy(42), ref1)
        nt.assert_equal(dummy(42, 43), ref2)


def test_memoize_does_not_cache_kwargs():

    global ctr
    ctr = 0

    @nu.memoize
    def dummy(x, y=42):
        global ctr
        ctr += 1
        return ctr

    for i in range(10):
        nt.assert_equal(dummy(42, y=43), i + 1)


def test_deprecated():
    @nu.deprecated(msg='Hello')
    def dummy():
        pass

    with warnings.catch_warnings(record=True) as s:
        dummy()
        nt.ok_(len(s) > 0)
        nt.eq_(s[0].message.args[0], 'Call to deprecated function dummy. Hello')


def test_deprecated_module():
    with warnings.catch_warnings(record=True) as s:
        nu.deprecated_module('foo', msg='msg')
        nt.ok_(len(s) > 0)


def test_NeuromJSON():
    ex = {'zero': 0,
          'one': np.int64(1),
          'two': np.float32(2.0),
          'three': np.array([1, 2, 3])
          }
    output = json.dumps(ex, cls=nu.NeuromJSON)
    loaded = json.loads(output)
    nt.eq_(loaded,
           {'zero': 0,
            'one': 1,
            'two': 2.0,
            'three': [1, 2, 3]
            })

    enc = nu.NeuromJSON()
    nt.eq_(enc.default(ex['one']), 1)
    nt.eq_(enc.default(ex['two']), 2.0)

    nt.assert_raises(TypeError, enc.default, 0)
