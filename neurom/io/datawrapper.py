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

"""Fast neuron IO module."""

import logging
from collections import defaultdict, namedtuple

import numpy as np
from neurom.core.dataformat import COLS, POINT_TYPE, ROOT_ID

L = logging.getLogger(__name__)


TYPE, ID, PID = 0, 1, 2


class DataWrapper(object):
    """Class holding a raw data block and section information."""

    def __init__(self, data_block, fmt, sections=None):
        """Section Data Wrapper.

        data_block is np.array-like with the following columns:
            [X, Y, Z, R, TYPE, ID, P]
            X(float): x-coordinate
            Y(float): y-coordinate
            Z(float): z-coordinate
            R(float): radius
            TYPE(integer): one of the types described by POINT_TYPE
            ID(integer): unique integer given to each point, the `ROOT_ID` is -1
            P(integer): the ID of the parent

        Args:
            data_block: as defined above
            fmt: File format designation, eg: SWC
            sections: Already extracted sections, otherwise data_block will be used

        Notes:
            - there is no ordering constraint: a child can reference a parent ID that comes
              later in the block
            - there is no requirement that the IDs are dense
            - there is no upper bound on the number of rows with the same 'P'arent: in other
              words, multifurcations are allowed
        """
        self.data_block = data_block
        self.fmt = fmt
        # list of DataBlockSection
        self.sections = sections if sections is not None else _extract_sections(data_block)

    def neurite_root_section_ids(self):
        """Get the section IDs of the intitial neurite sections."""
        sec = self.sections
        return [i for i, ss in enumerate(sec)
                if ss.pid > -1 and (sec[ss.pid].ntype == POINT_TYPE.SOMA and
                                    ss.ntype != POINT_TYPE.SOMA)]

    def soma_points(self):
        """Get the soma points."""
        db = self.data_block
        return db[db[:, COLS.TYPE] == POINT_TYPE.SOMA]


def _merge_sections(sec_a, sec_b):
    """Merge two sections.

    Merges sec_a into sec_b and sets sec_a attributes to default
    """
    sec_b.ids = list(sec_a.ids) + list(sec_b.ids[1:])
    sec_b.ntype = sec_a.ntype
    sec_b.pid = sec_a.pid

    sec_a.ids = []
    sec_a.pid = -1
    sec_a.ntype = 0


def _section_end_points(structure_block, id_map):
    """Get the section end-points."""
    soma_idx = structure_block[:, TYPE] == POINT_TYPE.SOMA
    soma_ids = structure_block[soma_idx, ID]
    neurite_idx = structure_block[:, TYPE] != POINT_TYPE.SOMA
    neurite_rows = structure_block[neurite_idx, :]
    soma_end_pts = set(id_map[id_]
                       for id_ in soma_ids[np.in1d(soma_ids, neurite_rows[:, PID])])

    # end points have either no children or more than one
    # ie: leaf or multifurcation nodes
    n_children = defaultdict(int)
    for row in structure_block:
        n_children[row[PID]] += 1
    end_pts = set(i for i, row in enumerate(structure_block)
                  if n_children[row[ID]] != 1)

    return end_pts.union(soma_end_pts)


class DataBlockSection(object):
    """Sections ((ids), type, parent_id)."""
    def __init__(self, ids=None, ntype=0, pid=-1):
        """Initialize a DataBlockSection object."""
        self.ids = [] if ids is None else ids
        self.ntype = ntype
        self.pid = pid

    def __eq__(self, other):
        """Test for equality."""
        return (self.ids == other.ids and
                self.ntype == other.ntype and
                self.pid == other.pid)

    def __str__(self):
        """Return a string representation."""
        return ('%s: ntype=%s, pid=%s: n_ids=%d' %
                (self.__class__, self.ntype, self.pid, len(self.ids)))

    __repr__ = __str__


def _extract_sections(data_block):
    """Make a list of sections from an SWC-style data wrapper block."""
    structure_block = data_block[:, COLS.TYPE:COLS.COL_COUNT].astype(np.int)

    # SWC ID -> structure_block position
    id_map = {-1: -1}
    for i, row in enumerate(structure_block):
        id_map[row[ID]] = i

    # end points have either no children, more than one, or are the start
    # of a new gap
    sec_end_pts = _section_end_points(structure_block, id_map)

    # a 'gap' is when a section has part of it's segments interleaved
    # with those of another section
    gap_sections = set()

    sections = []

    def new_section():
        """A new_section."""
        sections.append(DataBlockSection())
        return sections[-1]

    curr_section = new_section()

    parent_section = {-1: -1}

    for row in structure_block:
        row_id = id_map[row[ID]]
        parent_id = id_map[row[PID]]
        if not curr_section.ids:
            # first in section point is parent
            curr_section.ids.append(parent_id)
            curr_section.ntype = row[TYPE]

        gap = parent_id != curr_section.ids[-1]

        # If parent is not the previous point, create a section end-point.
        # Else add the point to this section
        if gap:
            sec_end_pts.add(row_id)
        else:
            curr_section.ids.append(row_id)

        if row_id in sec_end_pts:
            parent_section[curr_section.ids[-1]] = len(sections) - 1
            # Parent-child discontinuity section
            if gap:
                curr_section = new_section()
                curr_section.ids.extend((parent_id, row_id))
                curr_section.ntype = row[TYPE]
                gap_sections.add(len(sections) - 2)
            elif row_id != len(data_block) - 1:
                # avoid creating an extra DataBlockSection for last row if it's a leaf
                curr_section = new_section()

    for sec in sections:
        # get the section parent ID from the id of the first point.
        if sec.ids:
            sec.pid = parent_section[sec.ids[0]]

        # join gap sections and "disable" first half
        if sec.pid in gap_sections:
            _merge_sections(sections[sec.pid], sec)

    # TODO find a way to remove empty sections.  Currently they are
    # required to maintain tree integrity.
    return sections


class BlockNeuronBuilder(object):
    """Helper to create DataWrapper for 'block' sections.

    This helps create a new DataWrapper when one already has 'blocks'
    (ie: contiguous points, forming all the segments) of a section, and they
    just need to connect them together based on their parent.

    Example:
        >>> builder = BlockNeuronBuilder()
        >>> builder.add_section(segment_id, parent_id, segment_type, points)
        ...
        >>> morph = builder.get_datawrapper()

    Note:
        This will re-number the IDs if they are not 'dense' (ie: have gaps)
    """
    BlockSection = namedtuple('BlockSection', 'parent_id section_type points')

    def __init__(self):
        """Initialize a BlockNeuronBuilder object."""
        self.sections = {}

    def add_section(self, id_, parent_id, section_type, points):
        """Add a section.

        Args:
            id_(int): identifying number of the section
            parent_id(int): identifying number of the parent of this section
            section_type(int): the section type as defined by POINT_TYPE
            points: an array of [X, Y, Z, R]
        """
        # L.debug('Adding section %d, with parent %d, of type: %d with count: %d',
        #         id_, parent_id, section_type, len(points))
        assert id_ not in self.sections, 'id %s already exists in sections' % id_
        self.sections[id_] = BlockNeuronBuilder.BlockSection(parent_id, section_type, points)

    def _make_datablock(self):
        """Make a data_block and sections list as required by DataWrapper."""
        section_ids = sorted(self.sections)

        # create all insertion id's, this needs to be done ahead of time
        # as some of the children may have a lower id than their parents
        id_to_insert_id = {}
        row_count = 0
        for section_id in section_ids:
            row_count += len(self.sections[section_id].points)
            id_to_insert_id[section_id] = row_count - 1

        datablock = np.empty((row_count, COLS.COL_COUNT), dtype=np.float)
        datablock[:, COLS.ID] = np.arange(len(datablock))
        datablock[:, COLS.P] = datablock[:, COLS.ID] - 1

        sections = []
        insert_index = 0
        for id_ in section_ids:
            sec = self.sections[id_]
            points, section_type, parent_id = sec.points, sec.section_type, sec.parent_id

            idx = slice(insert_index, insert_index + len(points))
            datablock[idx, COLS.XYZR] = points
            datablock[idx, COLS.TYPE] = section_type
            datablock[idx.start, COLS.P] = id_to_insert_id.get(parent_id, ROOT_ID)
            sections.append(DataBlockSection(idx, section_type, parent_id))
            insert_index = idx.stop

        return datablock, sections

    def _check_consistency(self):
        """See if the sections have obvious errors."""
        type_count = defaultdict(int)
        for _, section in sorted(self.sections.items()):
            type_count[section.section_type] += 1

        if type_count[POINT_TYPE.SOMA] != 1:
            L.info('Have %d somas, expected 1', type_count[POINT_TYPE.SOMA])

    def get_datawrapper(self, file_format='BlockNeuronBuilder', data_wrapper=DataWrapper):
        """Returns a DataWrapper."""
        self._check_consistency()
        datablock, sections = self._make_datablock()
        return data_wrapper(datablock, file_format, sections)
