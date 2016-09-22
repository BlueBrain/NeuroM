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

'''Reader for Neurolucida .ASC files, v3, reversed engineered from looking at output from
Neuroludica
'''

import logging
import warnings

import numpy as np

from neurom.core.dataformat import COLS, POINT_TYPE
from .datawrapper import DataWrapper


WANTED_SECTIONS = {
    'CellBody': POINT_TYPE.SOMA,
    'Axon': POINT_TYPE.AXON,
    'Dendrite': POINT_TYPE.BASAL_DENDRITE,
    'Apical': POINT_TYPE.APICAL_DENDRITE,
}
UNWANTED_SECTION_NAMES = [
    # Meta-data?
    'Closed', 'Color', 'FillDensity', 'GUID', 'ImageCoords', 'MBFObjectType',
    'Marker', 'Name', 'Resolution', 'Set',
    # Marker names?
    'Asterisk', 'Cross', 'Dot', 'DoubleCircle', 'FilledCircle', 'FilledDownTriangle',
    'FilledSquare', 'FilledStar', 'FilledUpTriangle', 'FilledUpTriangle', 'Flower',
    'Flower2', 'OpenCircle', 'OpenDiamond', 'OpenDownTriangle', 'OpenSquare', 'OpenStar',
    'OpenUpTriangle', 'Plus', 'ShadedStar', 'Splat', 'TriStar',
]
UNWANTED_SECTIONS = dict([(name, True) for name in UNWANTED_SECTION_NAMES])
L = logging.getLogger(__name__)


def _match_section(section, match):
    '''checks whether the `type` of section is in the `match` dictionary

    Works around the unknown ordering of s-expressions in each section.
    For instance, the `type` is the 3-rd one in for CellBodies
        ("CellBody"
         (Color Yellow)
         (CellBody)
         (Set "cell10")
        )

    Returns:
        value associated with match[section_type], None if no match
    '''
    # TODO: rewrite this so it is more clear, and handles sets & dictionaries for matching
    for i in range(5):
        if i >= len(section):
            return None
        elif isinstance(section[i], (str, unicode)) and section[i] in match:
            return match[section[i]]
    return None


def _get_tokens(morph_fd):
    '''split a file-like into tokens: split on whitespace

    Note: this also strips newlines and comments
    '''
    for line in morph_fd.readlines():
        line = line.rstrip()   # remove \r\n
        line = line.split(';', 1)[0]  # strip comments
        squash_token = []  # quoted strings get squashed into one token
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


def _parse_section(token_iter):
    '''take a stream of tokens, and create the tree structure that is defined
    by the s-expressions
    '''
    sexp = []
    for token in token_iter:
        if '(' == token:
            new_sexp = _parse_section(token_iter)
            if not _match_section(new_sexp, UNWANTED_SECTIONS):
                sexp.append(new_sexp)
        elif ')' == token:
            return sexp
        else:
            sexp.append(token)
    return sexp


def _parse_sections(morph_fd):
    '''returns array of all the sections that exist

    The format is nested lists that correspond to the s-expressions
    '''
    sections = []
    token_iter = _get_tokens(morph_fd)
    for token in token_iter:
        if '(' == token:  # find top-level sections
            section = _parse_section(token_iter)
            if not _match_section(section, UNWANTED_SECTIONS):
                sections.append(section)
    return sections


def _flatten_subsection(subsection, _type, offset, parent):
    '''Flatten a subsection from its nested version

    Args:
        subsection: Nested subsection as produced by _parse_section, except one level in
        _type: type of section, ie: AXON, etc
        parent: first element has this as it's parent
        offset: position in the final array of the first element

    Returns:
        Generator of values corresponding to [X, Y, Z, R, TYPE, ID, PARENT_ID]
    '''
    for row in subsection:
        # TODO: Figure out what these correspond to in neurolucida
        if row in ('Low', 'Generated', 'High', ):
            continue
        elif isinstance(row[0], (str, unicode)):
            if 4 == len(row):
                yield (float(row[0]), float(row[1]), float(row[2]), float(row[3]) / 2.,
                       _type, offset, parent)
                parent = offset
                offset += 1
        elif isinstance(row[0], list):
            split_parent = offset - 1
            start_offset = 0

            slices = []
            start = 0
            for i, value in enumerate(row):
                if '|' == value:
                    slices.append(slice(start + start_offset, i))
                    start = i + 1
            slices.append(slice(start + start_offset, len(row)))

            for split_slice in slices:
                for _row in _flatten_subsection(row[split_slice], _type, offset,
                                                split_parent):
                    offset += 1
                    yield _row


def _extract_section(section):
    '''Find top level sections, and get their flat contents, and append them all

    Returns a numpy array with the row format:
        [X, Y, Z, R, TYPE, ID, PARENT_ID]

    Note: PARENT_ID starts at -1 for soma and 0 for neurites
    '''
    # try and detect type
    _type = WANTED_SECTIONS.get(section[0][0], None)

    start = 1
    # CellBody often has [['"CellBody"'], ['CellBody'] as its first two elements
    if _type is None:
        _type = WANTED_SECTIONS.get(section[1][0], None)
        if _type is None:  # can't determine the type
            return None
        start = 2

    parent = -1 if _type == POINT_TYPE.SOMA else 0
    subsection_iter = _flatten_subsection(section[start:], _type, offset=0,
                                          parent=parent)

    ret = np.array([row for row in subsection_iter])
    return ret


def _sections_to_raw_data(sections):
    '''convert list of sections into the `raw_data` format used in neurom

    This finds the soma, and attaches the neurites
    '''
    soma = None
    neurites = []
    for section in sections:
        neurite = _extract_section(section)
        if neurite is None:
            continue
        elif neurite[0][COLS.TYPE] == POINT_TYPE.SOMA:
            assert soma is None, 'Multiple somas defined in file'
            soma = neurite
        else:
            neurites.append(neurite)
    assert soma is not None, 'No soma found'

    total_length = len(soma) + sum(len(neurite) for neurite in neurites)
    ret = np.zeros((total_length, 7,), dtype=np.float64)
    pos = len(soma)
    ret[0:pos, :] = soma

    for neurite in neurites:
        end = pos + len(neurite)
        ret[pos:end, :] = neurite
        ret[pos:end, COLS.P] += pos
        ret[pos:end, COLS.ID] += pos
        # TODO: attach the neurite at the closest point on the soma
        ret[pos, COLS.P] = len(soma) - 1
        pos = end

    return ret


def read(morph_file, data_wrapper=DataWrapper):
    '''return a 'raw_data' np.array with the full neuron, and the format of the file
    suitable to be wrapped by DataWrapper
    '''

    msg = ('This is an experimental reader. '
           'There are no guarantees regarding ability to parse '
           'Neurolucida .asc files or correctness of output.')

    warnings.warn(msg)
    L.warning(msg)

    with open(morph_file) as morph_fd:
        sections = _parse_sections(morph_fd)
    raw_data = _sections_to_raw_data(sections)
    return data_wrapper(raw_data, 'NL-ASCII')
