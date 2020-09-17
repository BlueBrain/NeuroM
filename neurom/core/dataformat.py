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

"""Data format definitions."""
import warnings


_COL_COUNT = 7


class _COLS(object):
    """A class for internal columns."""
    TYPE, ID, P = 4, 5, 6


class _PublicColumns(object):
    """Column labels for internal data representation."""
    COL_COUNT = _COL_COUNT
    (X, Y, Z, R) = range(4)
    XY = slice(0, 2)
    XZ = slice(0, 3, 2)
    YZ = slice(1, 3)
    XYZ = slice(0, 3)
    XYZR = slice(0, 4)

    @property
    def TYPE(self):
        warnings.warn('Using _COLS.TYPE is now deprecated. '
                      'Please consider using "section.type" to get the type of a section.')
        return _COLS.TYPE

    @property
    def ID(self):
        warnings.warn('Using _COLS.ID is now deprecated')
        return _COLS.ID

    @property
    def P(self):
        warnings.warn('Using _COLS.P is now deprecated')
        return _COLS.P


COLS = _PublicColumns()


class POINT_TYPE(object):
    """Point types.

    These follow SWC specification.
    """
    (UNDEFINED, SOMA, AXON, BASAL_DENDRITE, APICAL_DENDRITE,
     FORK_POINT, END_POINT, CUSTOM) = range(8)

    NEURITES = (AXON, BASAL_DENDRITE, APICAL_DENDRITE)


ROOT_ID = -1
