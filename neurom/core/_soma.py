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

'''Soma classes and functions'''
from neurom.morphmath import average_points_dist
from neurom.core.dataformat import COLS
from neurom.exceptions import SomaError
import numpy as np


class SOMA_TYPE(object):
    '''Enumeration holding soma types

    * Type SinglePoint: single point at centre
    * Type ThreePoint: Three points on circumference of sphere
    * Type SimpleContour: More than three points
    * INVALID: Not satisfying any of the above
    '''
    INVALID, SinglePoint, ThreePoint, SimpleContour = xrange(4)

    @staticmethod
    def get_type(points):
        '''get the type of the soma'''
        npoints = len(points)
        return {0: SOMA_TYPE.INVALID,
                1: SOMA_TYPE.SinglePoint,
                3: SOMA_TYPE.ThreePoint,
                2: SOMA_TYPE.INVALID}.get(npoints, SOMA_TYPE.SimpleContour)


class Soma(object):
    '''Base class for a soma.

    Holds a list of raw data rows corresponding to soma points
    and provides iterator access to them.
    '''
    def __init__(self, points):

        self._points = points

    @property
    def center(self):
        '''Obtain the radius from the first stored point'''
        return self._points[0][:COLS.R]

    def iter(self):
        '''Iterator to soma contents'''
        return iter(self._points)

    @property
    def points(self):
        '''Get the set of (x, y, z, r) points this soma'''
        return self._points[:, 0:4]


class SomaSinglePoint(Soma):
    '''
    Type A: 1point soma
    Represented by a single point.
    '''
    def __init__(self, points):
        super(SomaSinglePoint, self).__init__(points)
        self.radius = points[0][COLS.R]

    def __str__(self):
        return 'SomaSinglePoint(%s) <center: %s, radius: %s>' % \
            (repr(self._points), self.center, self.radius)


class SomaThreePoint(Soma):
    '''
    Type B: 3point soma
    Represented by 3 points.

    Reference:
        http://neuromorpho.org/SomaFormat.html
        'The first point constitutes the center of the soma.
        An equivalent radius (rs) is computed as the average distance
        of the other two points.'
    '''
    def __init__(self, points):
        super(SomaThreePoint, self).__init__(points)
        self.radius = average_points_dist(points[0], (points[1], points[2]))

    def __str__(self):
        return 'SomaThreePoint(%s) <center: %s, radius: %s>' % \
            (repr(self._points), self.center, self.radius)


class SomaSimpleContour(Soma):
    '''
    Type C: multiple points soma
    Represented by a contour.

    The equivalent radius as the average distance to the center.

    Note: This doesn't currently check to see if the contour is in a plane. Also
    the radii of the points are not taken into account.
    '''
    def __init__(self, points):
        super(SomaSimpleContour, self).__init__(points)
        points = np.array(self._points)
        self.radius = average_points_dist(self.center, points[:, :COLS.R])

    @property
    def center(self):
        '''Obtain the center from the average of all points'''
        points = np.array(self._points)
        return np.mean(points[:, :COLS.R], axis=0)

    def __str__(self):
        return 'SomaSimpleContour(%s) <center: %s, radius: %s>' % \
            (repr(self._points), self.center, self.radius)


def make_soma(points, soma_check=None):
    '''Make a soma object from a set of points

    Infers the soma type (SomaSinglePoint, SomaThreePoint or SomaSimpleContour) from the points.

    Parameters:
        points: collection of points forming a soma.
        soma_check: optional validation function applied to points. Should
        raise a SomaError if points not valid.

    Raises:
        SomaError if no soma points found, points incompatible with soma, or
        if soma_check(points) fails.
    '''

    if soma_check:
        soma_check(points)

    stype = SOMA_TYPE.get_type(points)
    if stype == SOMA_TYPE.INVALID:
        raise SomaError('Invalid soma points')

    return {SOMA_TYPE.SinglePoint: SomaSinglePoint,
            SOMA_TYPE.ThreePoint: SomaThreePoint,
            SOMA_TYPE.SimpleContour: SomaSimpleContour}[stype](points)
