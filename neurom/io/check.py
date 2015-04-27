'''Module with consistency/validity checks for raw data  blocks'''
import numpy as np
from neurom.core.dataformat import ROOT_ID
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE


def has_sequential_ids(raw_data):
    '''Check that IDs are increasing and consecutive

    returns tuple (bool, list of IDs that are not consecutive
    with their predecessor)
    '''
    ids = raw_data.get_col(COLS.ID)
    steps = [int(j) for (i, j) in zip(ids, ids[1:]) if int(j - i) != 1]
    return len(steps) == 0, steps


def has_soma(raw_data):
    '''Checks if the TYPE column of raw data block has
    an element of type soma'''
    return POINT_TYPE.SOMA in raw_data.get_col(COLS.TYPE)


def has_all_finite_radius_neurites(raw_data):
    '''Check that all points with neurite type have a finite radius

    return: tuple of (bool, [IDs of neurite points with zero radius])
    '''
    bad_points = list()
    for row in raw_data.iter_row():
        if row[COLS.TYPE] in POINT_TYPE.NEURITES and row[COLS.R] == 0.0:
            bad_points.append(int(row[COLS.ID]))

    return len(bad_points) == 0, bad_points


def is_neurite_segment(segment):
    '''Check that both points in a segment are of neurite type

    argument:
        segment: pair of raw data rows representing a segment

    return: true if both have neurite type
    '''
    return (segment[0][COLS.TYPE] in POINT_TYPE.NEURITES and
            segment[1][COLS.TYPE] in POINT_TYPE.NEURITES)


def has_all_finite_length_segments(raw_data):
    '''Check that all segments of neurite type have a finite length

    return: tuple of (bool, [(pid, id) for zero-length segments])
    '''
    bad_segments = list()
    for row in raw_data.iter_row():
        idx = int(row[COLS.ID])
        pid = int(row[COLS.P])
        if pid != ROOT_ID:
            prow = raw_data.get_row(pid)
            if (is_neurite_segment((prow, row)) and
                    np.all(row[COLS.X: COLS.R] == prow[COLS.X: COLS.R])):
                bad_segments.append((pid, idx))
    return len(bad_segments) == 0, bad_segments
