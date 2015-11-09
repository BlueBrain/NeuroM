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

'''
Python module of NeuroM to check neuronal trees.
'''

import numpy as np
from neurom.core.tree import ipreorder
from neurom.core.dataformat import COLS
from neurom.analysis.morphtree import principal_direction_extent


def is_monotonic(tree, tol):
    '''Check if tree is monotonic, i.e. if each child has smaller or
        equal diameters from its parent

        Arguments:
            tree : tree object
            tol: numerical precision
    '''

    for node in ipreorder(tree):

        if node.parent is not None:

            if node.value[COLS.R] > node.parent.value[COLS.R] + tol:

                return False

    return True


def is_flat(tree, tol, method='tolerance'):
    '''Check if neurite is flat using the given method

        Input

            tree : the tree object

            tol : tolerance

            method : the method of flatness estimation.
            'tolerance' returns true if any extent of the tree
            is smaller than the given tolerance
            'ratio' returns true if the ratio of the smallest directions
            is smaller than tol. e.g. [1,2,3] -> 1/2 < tol

        Returns

            True if it is flat

    '''

    ext = principal_direction_extent(tree)

    if method == 'ratio':

        sorted_ext = np.sort(ext)
        return sorted_ext[0] / sorted_ext[1] < float(tol)

    else:

        return any(ext < float(tol))
