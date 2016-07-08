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

'''Section functions and functional tools'''

from itertools import izip
from neurom.core.tree import iupstream
from neurom.analysis import morphmath as mm


def map_sum_segments(fun, section):
    '''Map function to segments in section and sum the result'''
    return sum(fun(s) for s in izip(section[:-1], section[1:]))


def section_path_length(section):
    '''Path length from section to root'''
    return sum(mm.section_length(s.value) for s in iupstream(section))


def section_volume(section):
    '''Volume of a section'''
    return map_sum_segments(mm.segment_volume, section)


def section_area(section):
    '''Surface area of a section'''
    return map_sum_segments(mm.segment_area, section)


def section_tortuosity(section):
    '''Tortuosity of a section

    The tortuosity is defined as the ratio of the path length of a section
    and the euclidian distnce between its end points.

    The path length is the sum of distances between consecutive points.
    '''
    return mm.section_length(section) / mm.point_dist(section[-1], section[0])


def branch_order(section):
    '''Branching order of a tree section

    The branching order is defined as the depth of the tree section.

    Note:
        The first level has branch order 1.
    '''
    return sum(1 for _ in iupstream(section)) - 1


def section_radial_distance(section, origin):
    '''Return the radial distances of a tree section to a given origin point

    The radial distance is the euclidian distance between the
    end-point point of the section and the origin point in question.

    Parameters:
        section: neurite section object
        origin: point to which distances are measured. It must have at least 3\
            components. The first 3 components are (x, y, z).
    '''
    return mm.point_dist(section.value[-1], origin)
