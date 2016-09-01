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

'''Basic functions and iterators for neuron neurite point morphometrics

'''
import functools
from neurom.core import Tree
from .core import iter_neurites
from neurom.core.dataformat import COLS
from neurom.utils import deprecated_module

deprecated_module(__name__)

iter_type = Tree.ipreorder


def point_function(as_tree=False):
    '''Decorate a point function such that it can be used in neurite iteration

    Parameters:
        as_tree: specifies whether the function argument is a point of trees\
            or elements
    '''
    def _point_function(fun):
        '''Decorate a function with an iteration type member'''
        @functools.wraps(fun)
        def _wrapper(point):
            '''Simply pass arguments to wrapped function'''
            if not as_tree:
                point = point.value
            return fun(point)

        _wrapper.iter_type = iter_type
        return _wrapper

    return _point_function


@point_function(as_tree=True)
def identity(point):
    '''Hack to bind iteration type to do-nothing function'''
    return point


@point_function(as_tree=False)
def radius(point):
    '''Get the radius of a point'''
    return point[COLS.R]


def count(neuron):
    """
    Return number of segments in neuron or population
    """
    return sum(1 for _ in iter_neurites(neuron, identity))
