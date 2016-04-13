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

'''Basic functions and iterators for neuron neurite segment morphometrics

'''
import functools
from neurom.core import tree as tr
from neurom.core.dataformat import COLS
from neurom import iter_neurites
import neurom.analysis.morphmath as mm


iter_type = tr.isegment


def segment_function(as_tree=False):
    '''Decorate a segment function such that it can be used in neurite iteration

    Parameters:
        as_tree: specifies whether the function argument is a segment of trees\
            or elements
    '''
    def _segment_function(fun):
        '''Decorate a function with an iteration type member'''
        @functools.wraps(fun)
        def _wrapper(segment):
            '''Simply pass arguments to wrapped function'''
            if not as_tree:
                segment = (segment[0].value, segment[-1].value)
            return fun(segment)

        _wrapper.iter_type = tr.isegment
        return _wrapper

    return _segment_function


length = segment_function(as_tree=False)(mm.segment_length)
length2 = segment_function(as_tree=False)(mm.segment_length2)
radius = segment_function(as_tree=False)(mm.segment_radius)
volume = segment_function(as_tree=False)(mm.segment_volume)
area = segment_function(as_tree=False)(mm.segment_area)
taper_rate = segment_function(as_tree=False)(mm.segment_taper_rate)
x_coordinate = segment_function(as_tree=False)(mm.segment_x_coordinate)
y_coordinate = segment_function(as_tree=False)(mm.segment_y_coordinate)
z_coordinate = segment_function(as_tree=False)(mm.segment_z_coordinate)


@segment_function(as_tree=True)
def identity(segment):
    '''Hack to bind iteration type to do-nothing function'''
    return segment


def cross_section_at_fraction(segment, fraction):
    ''' Computes the point p along the line segment that connects the
    two ends of a segment p1p2 where |p1p| = fraction * |p1p2| along
    with the respective radius.
    Args:
        fraction: float between 0. and 1.

    Returns: tuple
        The 3D coordinates of the aforementioned point,
        Its respective radius
    '''
    return (mm.linear_interpolate(segment[0].value, segment[1].value, fraction),
            mm.interpolate_radius(segment[0].value[COLS.R], segment[1].value[COLS.R], fraction))


def radial_dist(pos):
    '''Return a function that calculates radial distance for a segment

    Radial distance calculater WRT pos
    '''
    @segment_function(as_tree=False)
    def _rad_dist(segment):
        '''Capture pos and invoke radial distance calculation'''
        return mm.segment_radial_dist(segment, pos)

    return _rad_dist


def count(neuron):
    """
    Return number of segments in neuron or population
    """
    return sum(1 for _ in iter_neurites(neuron, identity))
