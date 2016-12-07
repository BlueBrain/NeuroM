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

'''Fast neuron IO module'''

import logging

from collections import defaultdict, namedtuple
from neurom.core.dataformat import POINT_TYPE, COLS, ROOT_ID

import numpy as np

L = logging.getLogger(__name__)


class DataWrapper(object):
    '''Class holding a raw data block and section information'''

    def __init__(self, data_block, fmt, sections=None):
        '''Section Data Wrapper

        data_block is np.array-like with the following columns:
            [X, Y, Z, R, TYPE, ID, P]
            X(float): x-coordinate
            Y(float): y-coordinate
            Z(float): z-coordinate
            R(float): radius
            TYPE(integer): one of the types described by POINT_TYPE
            ID(integer): unique integer given to each point, the `ROOD_ID` is -1
            P(integer): the ID of the parent

        Notes:
            - there is no ordering constraint: a child can reference a parent ID that comes
              later in the block
            - there is no requirement that the IDs are dense
            - there is no upper bound on the number of rows with the same 'P'arent: in other
              words, multifurcations are allowed
        '''
        self.data_block = data_block
        self.fmt = fmt
        self.sections = sections if sections is not None else _extract_sections(data_block)

    def neurite_root_section_ids(self):
        '''Get the section IDs of the intitial neurite sections'''
        sec = self.sections
        return [i for i, ss in enumerate(sec)
                if ss.pid > -1 and (sec[ss.pid].ntype == POINT_TYPE.SOMA and
                                    ss.ntype != POINT_TYPE.SOMA)]

    def soma_points(self):
        '''Get the soma points'''
        db = self.data_block
        return db[db[:, COLS.TYPE] == POINT_TYPE.SOMA]


def _merge_sections(sec_a, sec_b):
    '''Merge two sections

    Merges sec_a into sec_b and sets sec_a attributes to default
    '''
    sec_b.ids = list(sec_a.ids) + list(sec_b.ids[1:])
    sec_b.ntype = sec_a.ntype
    sec_b.pid = sec_a.pid

    sec_a.ids = []
    sec_a.pid = -1
    sec_a.ntype = 0


def _section_end_points(data_block, id_map):
    '''Get the section end-points '''
    def _is_soma_neurite_break(idx):
        '''determine if idx is the index of the first non-soma point'''
        row = data_block[idx]
        pid = id_map[int(row[COLS.P])]
        if row[COLS.P] == ROOT_ID:
            return False

        return (row[COLS.TYPE] != POINT_TYPE.SOMA and
                data_block[pid][COLS.TYPE] == POINT_TYPE.SOMA)

    soma_end_pts = set()

    # number of children per point
    n_children = defaultdict(int)
    for i, row in enumerate(data_block):
        n_children[int(row[COLS.P])] += 1
        if _is_soma_neurite_break(i):
            soma_end_pts.add(id_map[int(data_block[i][COLS.P])])

    # end points have either no children or more than one
    end_pts = set(i for i, row in enumerate(data_block)
                  if n_children[row[COLS.ID]] != 1)

    return end_pts.union(soma_end_pts)


class Section(object):
    '''sections ((ids), type, parent_id)'''
    def __init__(self, ids=None, ntype=0, pid=-1):
        self.ids = [] if ids is None else ids
        self.ntype = ntype
        self.pid = pid

    def __eq__(self, other):
        return (self.ids == other.ids and
                self.ntype == other.ntype and
                self.pid == other.pid)


def _extract_sections(data_block):
    '''Make a list of sections from an SWC-style data wrapper block'''

    # get SWC ID to array position map
    id_map = {-1: -1}
    for i, r in enumerate(data_block):
        id_map[int(r[COLS.ID])] = i

    # end points have either no children or more than one
    sec_end_pts = _section_end_points(data_block, id_map)

    # artificial discontinuity section IDs
    _gap_sections = set()

    _sections = [Section()]
    curr_section = _sections[-1]
    parent_section = {-1: -1}

    for row in data_block:
        row_id = id_map[int(row[COLS.ID])]
        parent_id = id_map[int(row[COLS.P])]
        if len(curr_section.ids) == 0:
            # first in section point is parent.
            curr_section.ids.append(parent_id)
            curr_section.ntype = int(row[COLS.TYPE])
        gap = parent_id != curr_section.ids[-1]
        # If parent is not the previous point, create
        # a section end-point. Else add the point
        # to this section
        if gap:
            sec_end_pts.add(row_id)
        else:
            curr_section.ids.append(row_id)

        if row_id in sec_end_pts:
            parent_section[curr_section.ids[-1]] = len(_sections) - 1
            _sections.append(Section())
            curr_section = _sections[-1]
            # Parent-child discontinuity section
            if gap:
                curr_section.ids.extend((parent_id, row_id))
                curr_section.ntype = int(row[COLS.TYPE])
                _gap_sections.add(len(_sections) - 2)

    for sec in _sections:
        # get the section parent ID from the id of the first point.
        if sec.ids:
            sec.pid = parent_section[sec.ids[0]]
        # join gap sections and "disable" first half
        if sec.pid in _gap_sections:
            _merge_sections(_sections[sec.pid], sec)

    # TODO find a way to remove empty sections.
    # Currently they are required to maintain
    # tree integrity.
    return _sections


COL_COUNT = 7
_, _, _, _, TYPE, ID, PARENT = range(COL_COUNT)
XYZR = slice(0, 4)


class BlockNeuronBuilder(object):
    '''Helper to create DataWrapper for 'block' sections

    This helps create a new DataWrapper when one already has 'blocks'
    (ie: contiguous points, forming all the segments) of a section, and they
    just need to connect them together based on their parent.

    Example:
        >>> builder = BlockNeuronBuilder()
        >>> builder.add_section(segment_id, parent_id, segment_type, points)
        ...
        >>> morph = builder.get_datawrapper()
    '''
    BlockSection = namedtuple('BlockSection', 'parent_id section_type points')

    def __init__(self):
        self.sections = {}

    def add_section(self, id_, parent_id, section_type, points):
        '''add a section

        Args:
            id_(int): identifying number of the section
            parent_id(int): identifying number of the parent of this section
            section_type(int): the section type as defined by POINT_TYPE
            points is an array of [X, Y, Z, R]'''
        # L.debug('Adding section %d, with parent %d, of type: %d with count: %d',
        #         id_, parent_id, section_type, len(points))
        assert id_ not in self.sections, 'id %s already exists in sections' % id_
        self.sections[id_] = BlockNeuronBuilder.BlockSection(parent_id, section_type, points)

    def _make_datablock(self):
        '''Make a data_block and sections list as required by DataWrapper'''
        section_ids = sorted(self.sections)

        # create all insertion id's, this needs to be done ahead of time
        # as some of the children may have a lower id than their parents
        id_to_insert_id = {}
        row_count = 0
        for section_id in section_ids:
            row_count += len(self.sections[section_id].points)
            id_to_insert_id[section_id] = row_count - 1

        datablock = np.empty((row_count, COL_COUNT), dtype=np.float)
        datablock[:, ID] = np.arange(len(datablock))
        datablock[:, PARENT] = datablock[:, ID] - 1

        sections = []
        insert_index = 0
        for id_ in section_ids:
            sec = self.sections[id_]
            points, section_type, parent_id = sec.points, sec.section_type, sec.parent_id

            idx = slice(insert_index, insert_index + len(points))
            datablock[idx, XYZR] = points
            datablock[idx, TYPE] = section_type
            datablock[idx.start, PARENT] = id_to_insert_id.get(parent_id, ROOT_ID)
            sections.append(Section(idx, section_type, parent_id))
            insert_index = idx.stop

        return datablock, sections

    def _check_consistency(self):
        '''see if the sections have obvious errors'''
        type_count = defaultdict(int)
        for _, section in sorted(self.sections.items()):
            type_count[section.section_type] += 1

        if type_count[POINT_TYPE.SOMA] != 1:
            L.info('Have %d somas, expected 1', type_count[POINT_TYPE.SOMA])

    def get_datawrapper(self, file_format='BlockNeuronBuilder', data_wrapper=DataWrapper):
        '''returns a DataWrapper'''
        self._check_consistency()
        datablock, sections = self._make_datablock()
        return data_wrapper(datablock, file_format, sections)
