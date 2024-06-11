# Copyright (c) 2020, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
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

"""Bifurcation point functions."""

import numpy as np

import neurom.features.section
from neurom import morphmath
from neurom.core.dataformat import COLS
from neurom.core.morphology import Section
from neurom.exceptions import NeuroMError


def _raise_if_not_bifurcation(section):
    n_children = len(section.children)
    if n_children != 2:
        raise NeuroMError(
            'A bifurcation point must have exactly 2 children, found {}'.format(n_children)
        )


def local_bifurcation_angle(bif_point):
    """Return the opening angle between two out-going sections in a bifurcation point.

    We first ensure that the input point has only two children.

    The bifurcation angle is defined as the angle between the first non-zero
    length segments of a bifurcation point.
    """

    def skip_0_length(sec):
        """Return the first point with non-zero distance to first point."""
        p0 = sec[0]
        cur = sec[1]
        for i, p in enumerate(sec[1:]):
            if not np.all(p[: COLS.R] == p0[: COLS.R]):
                cur = sec[i + 1]
                break

        return cur

    _raise_if_not_bifurcation(bif_point)

    ch0, ch1 = (
        skip_0_length(bif_point.children[0].points),
        skip_0_length(bif_point.children[1].points),
    )

    return morphmath.angle_3points(bif_point.points[-1], ch0, ch1)


def remote_bifurcation_angle(bif_point):
    """Return the opening angle between two out-going sections in a bifurcation point.

    We first ensure that the input point has only two children.

    The angle is defined as between the bifurcation point and the
    last points in the out-going sections.
    """
    _raise_if_not_bifurcation(bif_point)

    return morphmath.angle_3points(
        bif_point.points[-1], bif_point.children[0].points[-1], bif_point.children[1].points[-1]
    )


def bifurcation_partition(bif_point, iterator_type=Section.ipreorder):
    """Calculate the partition at a bifurcation point.

    We first ensure that the input point has only two children.

    The number of nodes in each child tree is counted. The partition is
    defined as the ratio of the largest number to the smallest number.
    """
    _raise_if_not_bifurcation(bif_point)

    n, m = partition_pair(bif_point, iterator_type=iterator_type)

    return max(n, m) / min(n, m)


def partition_asymmetry(bif_point, uylings=False, iterator_type=Section.ipreorder):
    """Calculate the partition asymmetry at a bifurcation point.

    By default partition asymmetry is defined as in https://www.ncbi.nlm.nih.gov/pubmed/18568015.
    However if ``uylings=True`` is set then
    https://jvanpelt.nl/papers/Uylings_Network_13_2002_397-414.pdf is used.

    The number of nodes in each child tree is counted. The partition
    is defined as the ratio of the absolute difference and the sum
    of the number of bifurcations in the two child subtrees
    at each branch point.
    """
    _raise_if_not_bifurcation(bif_point)

    n, m = partition_pair(bif_point, iterator_type=iterator_type)

    if n == m == 1:
        # By definition the asymmetry A(1, 1) is zero
        return 0.0

    c = 2.0 if uylings else 0.0

    return abs(n - m) / abs(n + m - c)


def partition_pair(bif_point, iterator_type=Section.ipreorder):
    """Calculate the partition pairs at a bifurcation point.

    The number of nodes in each child tree is counted. The partition
    pairs is the number of bifurcations in the two child subtrees
    at each branch point.
    """
    return (
        float(len(list(iterator_type(bif_point.children[0])))),
        float(len(list(iterator_type(bif_point.children[1])))),
    )


def sibling_ratio(bif_point, method='first'):
    """Calculate the sibling ratio of a bifurcation point.

    The sibling ratio is the ratio between the diameters of the
    smallest and the largest child. It is a real number between
    0 and 1. Method argument allows one to consider mean diameters
    along the child section instead of diameter of the first point.
    """
    _raise_if_not_bifurcation(bif_point)

    if method == 'first':
        # the first point is the same as the parent last point
        n = bif_point.children[0].points[1, COLS.R]
        m = bif_point.children[1].points[1, COLS.R]
    elif method == 'mean':
        n = neurom.features.section.section_mean_radius(bif_point.children[0])
        m = neurom.features.section.section_mean_radius(bif_point.children[1])
    else:
        raise ValueError('Please provide a valid method for sibling ratio, found %s' % method)

    return min(n, m) / max(n, m)


def diameter_power_relation(bif_point, method='first'):
    """Calculate the diameter power relation at a bifurcation point.

    The diameter power relation is defined in https://www.ncbi.nlm.nih.gov/pubmed/18568015

    This quantity gives an indication of how far the branching is from
    the Rall ratio

    diameter_power_relation==1 means perfect Rall ratio
    """
    _raise_if_not_bifurcation(bif_point)

    if method == 'first':
        # the first point is the same as the parent last point
        d_child = bif_point.points[-1, COLS.R]
        d_child1 = bif_point.children[0].points[1, COLS.R]
        d_child2 = bif_point.children[1].points[1, COLS.R]
    elif method == 'mean':
        d_child = neurom.features.section.section_mean_radius(bif_point)
        d_child1 = neurom.features.section.section_mean_radius(bif_point.children[0])
        d_child2 = neurom.features.section.section_mean_radius(bif_point.children[1])
    else:
        raise ValueError('Please provide a valid method for sibling ratio, found %s' % method)

    return (d_child / d_child1) ** (1.5) + (d_child / d_child2) ** (1.5)


def downstream_pathlength_asymmetry(
    bif_point, normalization_length=1.0, iterator_type=Section.ipreorder
):
    """Calculates the downstream pathlength asymmetry at a bifurcation point.

    Args:
        bif_point: Bifurcation section.
        normalization_length: Constant to divide the result with.
        iterator_type: Iterator type that specifies how the two subtrees are traversed.

    Returns:
        The absolute difference between the downstream path distances of the two children, divided
        by the normalization length.
    """
    _raise_if_not_bifurcation(bif_point)
    return (
        abs(
            neurom.features.section.downstream_pathlength(
                bif_point.children[0], iterator_type=iterator_type
            )
            - neurom.features.section.downstream_pathlength(
                bif_point.children[1], iterator_type=iterator_type
            ),
        )
        / normalization_length
    )
