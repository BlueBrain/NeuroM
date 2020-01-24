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

'''Bifurcation point functions'''

import numpy as np
from neurom import morphmath
from neurom.exceptions import NeuroMError
from neurom.core.dataformat import COLS


def _raise_if_not_bifurcation(section):
    n_children = len(section.children)
    if n_children != 2:
        raise NeuroMError('A bifurcation point must have exactly 2 children, found {}'.format(
            n_children))


def local_bifurcation_angle(bif_point):
    '''Return the opening angle between two out-going sections
    in a bifurcation point

    We first ensure that the input point has only two children.

    The bifurcation angle is defined as the angle between the first non-zero
    length segments of a bifurcation point.
    '''
    def skip_0_length(sec):
        '''Return the first point with non-zero distance to first point'''
        p0 = sec[0]
        cur = sec[1]
        for i, p in enumerate(sec[1:]):
            if not np.all(p[:COLS.R] == p0[:COLS.R]):
                cur = sec[i + 1]
                break

        return cur

    _raise_if_not_bifurcation(bif_point)

    ch0, ch1 = (skip_0_length(bif_point.children[0].points),
                skip_0_length(bif_point.children[1].points))

    return morphmath.angle_3points(bif_point.points[-1], ch0, ch1)


def remote_bifurcation_angle(bif_point):
    '''Return the opening angle between two out-going sections
    in a bifurcation point

    We first ensure that the input point has only two children.

    The angle is defined as between the bifurcation point and the
    last points in the out-going sections.
    '''
    _raise_if_not_bifurcation(bif_point)

    return morphmath.angle_3points(bif_point.points[-1],
                                   bif_point.children[0].points[-1],
                                   bif_point.children[1].points[-1])


def bifurcation_partition(bif_point):
    '''Calculate the partition at a bifurcation point

    We first ensure that the input point has only two children.

    The number of nodes in each child tree is counted. The partition is
    defined as the ratio of the largest number to the smallest number.'''
    _raise_if_not_bifurcation(bif_point)

    n = float(sum(1 for _ in bif_point.children[0].ipreorder()))
    m = float(sum(1 for _ in bif_point.children[1].ipreorder()))
    return max(n, m) / min(n, m)


def partition_asymmetry(bif_point):
    '''Calculate the partition asymmetry at a bifurcation point
    as defined in https://www.ncbi.nlm.nih.gov/pubmed/18568015

    The number of nodes in each child tree is counted. The partition
    is defined as the ratio of the absolute difference and the sum
    of the number of bifurcations in the two daughter subtrees
    at each branch point.'''
    _raise_if_not_bifurcation(bif_point)

    n = float(sum(1 for _ in bif_point.children[0].ipreorder()))
    m = float(sum(1 for _ in bif_point.children[1].ipreorder()))
    if n == m:
        return 0.0
    return abs(n - m) / abs(n + m)


def partition_pair(bif_point):
    '''Calculate the partition pairs at a bifurcation point

    The number of nodes in each child tree is counted. The partition
    pairs is the number of bifurcations in the two daughter subtrees
    at each branch point.'''
    n = float(sum(1 for _ in bif_point.children[0].ipreorder()))
    m = float(sum(1 for _ in bif_point.children[1].ipreorder()))
    return (n, m)
