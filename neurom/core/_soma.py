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

"""Soma classes and functions."""
import math
import warnings

import numpy as np
from neurom import morphmath
from neurom.core.dataformat import COLS
from neurom.exceptions import SomaError


class Soma(object):
    """Base class for a soma.

    Holds a list of raw data rows corresponding to soma points
    and provides iterator access to them.
    """

    def __init__(self, points):
        """Initialize a Soma object."""
        self._points = points
        self.radius = 0

    @property
    def center(self):
        """Obtain the center from the first stored point."""
        return self._points[0][COLS.XYZ]

    def iter(self):
        """Iterator to soma contents."""
        return iter(self._points)

    @property
    def points(self):
        """Get the set of (x, y, z, r) points this soma."""
        return self._points[:, COLS.XYZR]

    @property
    def volume(self):
        """Gets soma volume assuming it is a sphere."""
        warnings.warn('Approximating soma volume by a sphere. {}'.format(self))
        return 4.0 / 3 * math.pi * self.radius ** 3


class SomaSinglePoint(Soma):
    """Type A: 1point soma.

    Represented by a single point.
    """

    def __init__(self, points):
        """Initialize a SomaSinglePoint object."""
        super().__init__(points)
        self.radius = points[0][COLS.R]

    def __str__(self):
        """Return a string representation."""
        return ('SomaSinglePoint(%s) <center: %s, radius: %s>' %
                (repr(self._points), self.center, self.radius))


class SomaCylinders(Soma):
    """Soma composed of cylinders (like in SWC).

    points describe the locations of the cylinder start/end points, with
    their respective radii, much like how neurites are described:

        ex::

                             /)
                            / o)
                    ______ /  )
                (|)      ) / /
                ( o )    o )/
                (|)_____ )

      Here we have a 'side-view', with each 'o' representing a point, and the
      radius is the height of a '|' character, and the ')' try and show the
      curvature of the cylinger

    Note: when, as in the case above, the cylinder center points don't lie
    in a line, then the overlap between cylinders isn't taken into account for
    the area calculation
    """

    def __init__(self, points):
        """Initialize a SomaCyliners object."""
        super().__init__(points)
        self.area = sum(morphmath.segment_area((p0, p1))
                        for p0, p1 in zip(points, points[1:]))
        self.radius = math.sqrt(self.area / (4. * math.pi))

    @property
    def center(self):
        """Obtain the center from the first stored point."""
        return self._points[0][COLS.XYZ]

    @property
    def volume(self):
        """Return the volume of soma."""
        return sum(morphmath.segment_volume((p0, p1))
                   for p0, p1 in zip(self.points, self.points[1:]))

    def __str__(self):
        """Return a string representation."""
        return ('SomaCylinders(%s) <center: %s, virtual radius: %s>' %
                (repr(self._points), self.center, self.radius))


class SomaNeuromorphoThreePointCylinders(SomaCylinders):
    """NeuroMorpho compatible soma.

    Reference:
        http://neuromorpho.org/SomaFormat.html

        Quote (Tue Feb 28 14:26:56 CET 2017):
        The first point constitutes the center of the soma, with coordinates (xs, ys,
        zs) corresponding to the average of all the points in the single contour. An
        equivalent radius (rs) is computed as the average distance of each point of the
        single contour from this center. The first point is assigned parent ID -1 as is
        standard for the root in the SWC format. The second and third soma points, as
        well as all starting points (roots) of dendritic and axonal arbors have this
        first point as the parent (parent ID 1). The x and z coordinates as well as the
        radius of both the second and third soma points are the same as that of the
        first point. The y coordinates of the second and third points are shifted by -rs
        and +rs, respectively (Figure 2). The surface area of this soma cylinder equals
        the surface area of a sphere of radius rs.
    """

    def __init__(self, points):
        """Initialize a SomaNeuromorphoThreePointCylinders object."""
        super().__init__(points)

        # X    Y     Z   R    P
        # xs ys      zs rs   -1
        # xs (ys-rs) zs rs    1
        # xs (ys+rs) zs rs    1

        r = points[0, COLS.R]
        # make sure the above invariant holds
        assert (np.isclose(r, points[1, COLS.R]) and np.isclose(r, points[2, COLS.R])), \
            'All radii must be the same'
        # only warn users about invalid format
        if r < 1e-5:
            warnings.warn('Zero radius for {}'.format(self))
        if not np.isclose(points[0, COLS.Y] - points[1, COLS.Y], r):
            warnings.warn(
                'The second point must be one radius below 0 on the y-plane for {}'.format(self))
        if not np.isclose(points[0, COLS.Y] - points[2, COLS.Y], -r):
            warnings.warn(
                'The third point must be one radius above 0 on the y-plane for {}'.format(self))
        h = morphmath.point_dist(points[1, COLS.XYZ], points[2, COLS.XYZ])
        self.area = 2.0 * math.pi * r * h  # ignores the 'end-caps' of the cylinder
        self.radius = math.sqrt(self.area / (4. * math.pi))

    @property
    def volume(self):
        """Return the volume of the soma."""
        return 2 * math.pi * self.radius ** 3

    def __str__(self):
        """Return a string representation."""
        return ('SomaNeuromorphoThreePointCylinders(%s) <center: %s, radius: %s>' %
                (repr(self._points), self.center, self.radius))


class SomaSimpleContour(Soma):
    """Type C: multiple points soma.

    Represented by a contour.

    The equivalent radius as the average distance to the center.

    Note: This doesn't currently check to see if the contour is in a plane. Also
    the radii of the points are not taken into account.
    """

    def __init__(self, points):
        """Initialize a SomaSimpleContour object."""
        super().__init__(points)
        points = np.array(self._points)
        self.radius = morphmath.average_points_dist(
            self.center, points[:, COLS.XYZ])

    @property
    def center(self):
        """Obtain the center from the average of all points."""
        points = np.array(self._points)
        return np.mean(points[:, COLS.XYZ], axis=0)

    def __str__(self):
        """Return a string representation."""
        return ('SomaSimpleContour(%s) <center: %s, radius: %s>' %
                (repr(self._points), self.center, self.radius))


# classes of somas
SOMA_CONTOUR = 'contour'
SOMA_CYLINDER = 'cylinder'


def _get_type(points, soma_class):
    """Get the type of the soma.

    Args:
        points: Soma points
        soma_class(str): one of 'contour' or 'cylinder' to specify the type
    """
    assert soma_class in (SOMA_CONTOUR, SOMA_CYLINDER)

    npoints = len(points)
    if soma_class == SOMA_CONTOUR:
        return {0: None,
                1: SomaSinglePoint,
                2: None}.get(npoints, SomaSimpleContour)

    if(npoints == 3 and
       points[0][COLS.P] == -1 and
       points[1][COLS.P] == 1 and
       points[2][COLS.P] == 1):
        warnings.warn('Using neuromorpho 3-Point soma')
        # NeuroMorpho is the main provider of morphologies, but they
        # with SWC as their default file format: they convert all
        # uploads to SWC.  In the process of conversion, they turn all
        # somas into their custom 'Three-point soma representation':
        #  http://neuromorpho.org/SomaFormat.html

        return SomaNeuromorphoThreePointCylinders

    return {0: None,
            1: SomaSinglePoint}.get(npoints, SomaCylinders)


def make_soma(points, soma_check=None, soma_class=SOMA_CONTOUR):
    """Make a soma object from a set of points.

    Infers the soma type (SomaSinglePoint, SomaSimpleContour)
    from the points and the 'soma_class'

    Arguments:
        points: collection of points forming a soma.
        soma_check: optional validation function applied to points. Should
        raise a SomaError if points not valid.
        soma_class(str): one of 'contour' or 'cylinder' to specify the type

    Raises:
        SomaError if no soma points found, points incompatible with soma, or
        if soma_check(points) fails.
    """
    if soma_check:
        soma_check(points)

    stype = _get_type(points, soma_class)

    if stype is None:
        raise SomaError('Invalid soma points')

    return stype(points)
