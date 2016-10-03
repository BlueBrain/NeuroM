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
from functools import wraps
from neurom import morphmath as mm


def section_fun(fun):
    '''Wrapper to extract points from section argument'''
    @wraps(fun)
    def _secfun(sec, **kwargs):
        '''Get points and forward to fun'''
        return fun(sec.points, **kwargs)

    return _secfun


def map_segments(fun, section):
    '''Map a function to segments in a section'''
    pts = section.points
    return list(fun(s) for s in izip(pts[:-1], pts[1:]))


def section_path_length(section):
    '''Path length from section to root'''
    return sum(mm.section_length(s.points) for s in section.iupstream())


def section_volume(section):
    '''Volume of a section'''
    return section.volume


def section_area(section):
    '''Surface area of a section'''
    return section.area


def section_tortuosity(section):
    '''Tortuosity of a section

    The tortuosity is defined as the ratio of the path length of a section
    and the euclidian distnce between its end points.

    The path length is the sum of distances between consecutive points.

    If the section contains less than 2 points, the value 1 is returned.
    '''
    pts = section.points
    return 1 if len(pts) < 2 else mm.section_length(pts) / mm.point_dist(pts[-1], pts[0])


def branch_order(section):
    '''Branching order of a tree section

    The branching order is defined as the depth of the tree section.

    Note:
        The first level has branch order 1.
    '''
    return sum(1 for _ in section.iupstream()) - 1


def section_radial_distance(section, origin):
    '''Return the radial distances of a tree section to a given origin point

    The radial distance is the euclidian distance between the
    end-point point of the section and the origin point in question.

    Parameters:
        section: neurite section object
        origin: point to which distances are measured. It must have at least 3\
            components. The first 3 components are (x, y, z).
    '''
    return mm.point_dist(section.points[-1], origin)


def section_meander_angles(section):
    '''Inter-segment opening angles in a section'''
    p = section.points
    return [mm.angle_3points(p[i - 1], p[i - 2], p[i])
            for i in xrange(2, len(p))]
