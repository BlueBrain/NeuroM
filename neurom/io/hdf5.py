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

"""Module for morphology HDF5 data loading.

Data is unpacked into a 2-dimensional raw data block:

    [X, Y, Z, R, TYPE, ID, PARENT_ID]


HDF5.V1 Input row format:
            points: [X, Y, Z, D] (ID is position)
            groups: [FIRST_POINT_ID, TYPE, PARENT_GROUP_ID]

There is one such row per measured point.
"""

from itertools import zip_longest

import h5py
import numpy as np

from .datawrapper import BlockNeuronBuilder, DataWrapper


def get_version(h5file):
    """Determine whether an HDF5 file is v1 or v2.

    Return: 'H5V1', 'H5V2' or None
    """
    if 'points' in h5file and 'structure' in h5file:
        return 'H5V1'
    if 'neuron1/structure' in h5file:
        return 'H5V2'
    return None


POINT_DIAMETER = 3
GPFIRST, GTYPE, GPID = range(3)  # groups or structure


def read(filename, remove_duplicates=False, data_wrapper=DataWrapper):
    """Read a file and return a `data_wrapper'd` data.

    * Tries to guess the format and the H5 version.
    * Unpacks the first block it finds out of ('repaired', 'unraveled', 'raw')

    Arguments:
        filename: path to file to be read
        remove_duplicates: boolean, If True removes duplicate points
            from the beginning of each section.
        data_wrapper: return class
    """
    with h5py.File(filename, mode='r') as h5file:
        version = get_version(h5file)
        if version == 'H5V1':
            points, groups = _unpack_v1(h5file)
        elif version == 'H5V2':
            stg = next(s for s in ('repaired', 'unraveled', 'raw')
                       if s in h5file['neuron1'])
            points, groups = _unpack_v2(h5file, stage=stg)

    if remove_duplicates:
        points, groups = _remove_duplicate_points(points, groups)

    neuron_builder = BlockNeuronBuilder()
    points[:, POINT_DIAMETER] /= 2  # Store radius, not diameter
    for id_, row in enumerate(zip_longest(groups,
                                          groups[1:, GPFIRST],
                                          fillvalue=len(points))):
        (point_start, section_type, parent_id), point_end = row
        neuron_builder.add_section(id_, int(parent_id), int(section_type),
                                   points[point_start:point_end])

    return neuron_builder.get_datawrapper(version, data_wrapper=data_wrapper)


def _remove_duplicate_points(points, groups):
    """Removes the duplicate points from the beginning of a section.

    If they are present in points-groups representation.

    Returns:
        points, groups with unique points.
    """
    group_initial_ids = groups[:, GPFIRST]

    to_be_reduced = np.zeros(len(group_initial_ids))
    to_be_removed = []

    for ig, g in enumerate(groups):
        iid, typ, pid = g[GPFIRST], g[GTYPE], g[GPID]
        # Remove first point from sections that are
        # not the root section, a soma, or a child of a soma
        if pid != -1 and typ != 1 and groups[pid][GTYPE] != 1:
            # Remove duplicate from list of points
            to_be_removed.append(iid)
            # Reduce the id of the following sections
            # in groups structure by one
            to_be_reduced[ig + 1:] += 1

    groups[:, GPFIRST] = groups[:, GPFIRST] - to_be_reduced
    points = np.delete(points, to_be_removed, axis=0)

    return points, groups


def _unpack_v1(h5file):
    """Unpack groups from HDF5 v1 file."""
    points = np.array(h5file['points'])
    groups = np.array(h5file['structure'])
    return points, groups


def _unpack_v2(h5file, stage):
    """Unpack groups from HDF5 v2 file."""
    points = np.array(h5file['neuron1/%s/points' % stage])
    # from documentation: The /neuron1/structure/unraveled reuses /neuron1/structure/raw
    groups_stage = stage if stage != 'unraveled' else 'raw'
    groups = np.array(h5file['neuron1/structure/%s' % groups_stage])
    stypes = np.array(h5file['neuron1/structure/sectiontype'])
    groups = np.hstack([groups, stypes])
    groups[:, [1, 2]] = groups[:, [2, 1]]
    return points, groups
