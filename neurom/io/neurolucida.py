# Copyright (c) 2016, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
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

'''Reader for Neurolucida .ASC files, v3, reverse engineered from looking at output from
Neuroludica
'''

import logging
import warnings
from neurom._compat import StringType

import numpy as np

from neurom.core.dataformat import POINT_TYPE
from .datawrapper import BlockNeuronBuilder


WANTED_SECTIONS = {
    'CellBody': POINT_TYPE.SOMA,
    'Axon': POINT_TYPE.AXON,
    'Dendrite': POINT_TYPE.BASAL_DENDRITE,
    'Apical': POINT_TYPE.APICAL_DENDRITE,
}
UNWANTED_SECTIONS = set([
    # Meta-data?
    'Closed', 'Color', 'FillDensity', 'GUID', 'ImageCoords', 'MBFObjectType',
    'Marker', 'Name', 'Resolution', 'Set', 'Sections',
    # Marker names?
    'Asterisk', 'Cross', 'Dot', 'DoubleCircle', 'FilledCircle', 'FilledDownTriangle',
    'FilledSquare', 'FilledStar', 'FilledUpTriangle', 'FilledUpTriangle', 'Flower',
    'Flower2', 'OpenCircle', 'OpenDiamond', 'OpenDownTriangle', 'OpenSquare', 'OpenStar',
    'OpenUpTriangle', 'Plus', 'ShadedStar', 'Splat', 'TriStar', 'CircleArrow', 'CircleCross',
    'FilledDiamond', 'MalteseCross', 'SnowFlake', 'TexacoStar', 'FilledQuadStar',
    'Circle1', 'Circle2', 'Circle3', 'Circle4', 'Circle5',
    'Circle6', 'Circle7', 'Circle8', 'Circle9',
])
L = logging.getLogger(__name__)


def _get_tokens(morph_fd):
    '''split a file-like into tokens: split on whitespace

    Note: this also strips comments and spines
    '''
    for line in morph_fd.readlines():
        line = line.split(';', 1)[0]  # strip comments
        squash_token = []  # quoted strings get squashed into one token, can be multi-line

        if '<(' in line:  # skip spines, which exist on a single line
            assert ')>' in line, 'Missing end of spine'
            continue  # pragma: no cover

        for token in line.replace('(', ' ( ').replace(')', ' ) ').split():
            if squash_token:
                squash_token.append(token)
                if token.endswith('"'):
                    token = ' '.join(squash_token)
                    squash_token = []
                    yield token
            elif token.startswith('"') and not token.endswith('"'):
                squash_token.append(token)
            else:
                yield token


def _consume_until_balanced_paren(token_iter, opening_count=1):
    '''Consume tokens until a opening_count close parens, taking into account balanced pairs'''
    opening_count = 1
    for token in token_iter:
        if token == ')':
            opening_count -= 1
        elif token == '(':
            opening_count += 1

        if opening_count == 0:
            break


def _parse_section(token_iter):
    '''extract from tokens the tree structure that is defined by the s-expressions

    Note: sections tagged as UNWANTED_SECTIONS are not returned
    '''
    sexp = []
    for token in token_iter:
        if '(' == token:
            sub_sexp = _parse_section(token_iter)
            if sub_sexp:
                sexp.append(sub_sexp)
        elif ')' == token:
            return sexp
        elif token in UNWANTED_SECTIONS:
            _consume_until_balanced_paren(token_iter)
            break
        else:
            sexp.append(token)
    return sexp


def _top_level_sections(morph_fd):
    '''yields the top level sections that exist

    The format is nested lists that correspond to the s-expressions
    '''
    token_iter = _get_tokens(morph_fd)
    for token in token_iter:
        if '(' == token:  # find top-level sections
            section = _parse_section(token_iter)
            if section:
                yield section


BLOCK_MARKERS = set(['Low', 'Generated', 'High', 'Normal', 'Incomplete',
                     'Midpoint', 'Origin', ])


def _extract_section_points(section):
    '''In section, extract all points in the section before a furcation point

    A furcation point is detected as a sub-list
    '''
    ret = []
    for row in section:
        if not isinstance(row[0], StringType):
            # probably a bifurcation point
            break
        elif isinstance(row, StringType):
            if row not in BLOCK_MARKERS:
                L.warning('Row: contains unknown block marker: %s', row)
            continue  # pragma: no cover

        assert len(row) in (4, 5, ), 'Point row contains more columns than 4 or 5: %s' % row

        if 5 == len(row) and 'S' != row[4][0]:
            L.warning('Only known usage of a fifth member is Sn, found: %s', row)

        ret.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]) / 2., ))

    return ret


def _find_furcations(rows):
    '''Neurolucida uses a '|' character for bifurcations'''
    furcations = []
    start_start = 0
    for i, value in enumerate(rows):
        if '|' == value:
            furcations.append(slice(start_start, i))
            start_start = i + 1
    furcations.append(slice(start_start, len(rows)))
    return furcations


def read_subsection(neuron_builder, id_, parent_id, section_type, subsection, parent_point=None):
    '''recursively extract each section within the section'''
    points = _extract_section_points(subsection)
    used_points = len(points)
    if parent_point is None:
        points = np.array(points)
    else:
        points = np.vstack((parent_point, points))

    neuron_builder.add_section(id_, parent_id, section_type, points)
    next_id = id_ + 1

    rest = subsection[used_points:]
    if rest and isinstance(rest[0], list):
        parent_point = points[-1]
        rest = rest[0]
        furcations = _find_furcations(rest)

        for split in furcations:
            subsection = rest[split]
            if not subsection or isinstance(subsection, StringType):
                continue  # pragma: no cover
            next_id = read_subsection(
                neuron_builder, next_id, id_, section_type, subsection, parent_point)

    return next_id


def read(morph_file):
    '''read Neurolucida file, returns a DataWrapper instance'''
    msg = ('This is an experimental reader. '
           'There are no guarantees regarding ability to parse '
           'Neurolucida .asc files or correctness of output.')
    warnings.warn(msg)

    neuron_builder = BlockNeuronBuilder()
    with open(morph_file) as morph_fd:
        id_ = 0
        for section in _top_level_sections(morph_fd):
            # try and detect type
            _type = WANTED_SECTIONS.get(section[0][0], None)
            start = 1

            # CellBody often has [['"CellBody"'], ['CellBody'] as its first two elements
            if _type is None:
                _type = WANTED_SECTIONS.get(section[1][0], None)
                start = 2

                if _type is None:  # can't determine the type, skip section
                    continue  # pragma: no cover

            # TODO: all neurites are connected at the 0 point of the soma, should probably
            # be the point closest to the neurite start
            parent_id = 0
            parent_point = None
            if _type == POINT_TYPE.SOMA:
                parent_id = -1
                parent_point = None

            id_ = read_subsection(
                neuron_builder, id_, parent_id, _type, section[start:], parent_point=parent_point)

    return neuron_builder.get_datawrapper('NL-ASCII')
