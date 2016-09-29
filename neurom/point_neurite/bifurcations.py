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

'''Basic functions and iterators for neuron neurite bifurcation point morphometrics

'''
import functools
from neurom.core import Tree
from . import point_tree as ptr
from . import treefunc as mt
from .core import iter_neurites
from neurom import morphmath as mm
from neurom.utils import deprecated_module


deprecated_module(__name__)

iter_type = Tree.ibifurcation_point


def bifurcation_point_function(as_tree=False):
    '''Decorate a bifurcation_point function such that it can be used in neurite iteration

    Parameters:
        as_tree: specifies whether the function argument is a\
            bifurcation point of trees or elements
    '''
    def _bifurcation_point_function(fun):
        '''Decorate a function with an iteration type member'''
        @functools.wraps(fun)
        def _wrapper(bifurcation_point):
            '''Simply pass arguments to wrapped function'''
            if not as_tree:
                bifurcation_point = bifurcation_point.value
            return fun(bifurcation_point)

        _wrapper.iter_type = iter_type
        return _wrapper

    return _bifurcation_point_function


local_angle = bifurcation_point_function(as_tree=True)(mt.local_bifurcation_angle)


@bifurcation_point_function(as_tree=True)
def identity(bifurcation_point):
    '''Hack to bind iteration type to do-nothing function'''
    return bifurcation_point


@bifurcation_point_function(as_tree=True)
def remote_angle(bifurcation_point):
    '''Calculate the remote bifurcation angle'''
    end_points = tuple(p for p in ptr.i_branch_end_points(bifurcation_point))
    return mm.angle_3points(bifurcation_point.value,
                            end_points[0].value,
                            end_points[1].value)


@bifurcation_point_function(as_tree=True)
def partition(bifurcation_point):
    '''Calculate the partition on each bif point
    '''
    n = float(mt.n_sections(bifurcation_point.children[0]))
    m = float(mt.n_sections(bifurcation_point.children[1]))
    return max(n, m) / min(n, m)


def count(neuron):
    """
    Return number of bifurcation points in neuron or population
    """
    return sum(1 for _ in iter_neurites(neuron, identity))
