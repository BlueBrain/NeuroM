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

''' Module for morphology SWC data loading

Data is unpacked into a 2-dimensional raw data block:

    [X, Y, Z, R, TYPE, ID, PARENT_ID]

SWC format:
    [ID, TYPE, X, Y, Z, R, PARENT_ID]

There is one such row per measured point.

'''
from functools import partial
from neurom.io.swc import read as _read
import numpy as np
from neurom.core.dataformat import COLS
from neurom.core.dataformat import ROOT_ID
from .datawrapper import DataWrapper


class SWCDataWrapper(DataWrapper):
    '''Specialization of DataWrapper for SWC data

    SWC data has looser requirements on the point IDs, so
    an ID to array index look-up table must be maintained.

    Index validity is checked and mappings performed before
    delegating to base class methods.
    '''
    def __init__(self, raw_data, fmt, _=None):
        super(SWCDataWrapper, self).__init__(raw_data, fmt)

        self._id_map = {}
        for i, row in enumerate(self.data_block):
            self._id_map[int(row[COLS.ID])] = i

        self._ids = np.array(self.data_block[:, COLS.ID],
                             dtype=np.int32).tolist()
        self._id_set = set(self._ids)

    def _get_idx(self, idx):
        ''' Apply global offset to an id'''
        return self._id_map[idx]

    def get_children(self, idx):
        ''' get list of ids of children of parent with id idx'''
        if idx in self._id_set or idx == ROOT_ID:
            return super(SWCDataWrapper, self).get_children(idx)

        raise LookupError('Invalid id: {0}'.format(idx))

    def get_parent(self, idx):
        '''get the parent of element with id idx'''
        if idx not in self._id_set:
            raise LookupError('Invalid id: {0}'.format(idx))

        idx = self._get_idx(idx)
        return super(SWCDataWrapper, self).get_parent(idx)

    def get_point(self, idx):
        '''Get point data for element idx'''
        idx = self._get_idx(idx)
        return super(SWCDataWrapper, self).get_point(idx)

    def get_row(self, idx):
        '''Get row from idx'''
        idx = self._get_idx(idx)
        return super(SWCDataWrapper, self).get_row(idx)


read = partial(_read, data_wrapper=SWCDataWrapper)
