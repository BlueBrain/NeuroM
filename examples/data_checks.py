'''Examples of basic data checks'''
import numpy as np
from neurom.io.readers import load_data
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


def finite_radius_neurites(raw_data):
    '''Check that points with neurite type have a finite radius

    return: tuple of (bool, [IDs of neurite points with zero radius])
    '''
    bad_points = list()
    for row in raw_data.iter_row():
        if row[COLS.TYPE] in POINT_TYPE.NEURITES and row[COLS.R] == 0.0:
            bad_points.append(int(row[COLS.ID]))

    return len(bad_points) == 0, bad_points


def has_finite_length_segments(raw_data):
    '''Check that all segments have a finite length

    return: tuple of (bool, [(pid, id) for zero-length segments])
    '''
    bad_segments = list()
    for row in raw_data.iter_row():
        idx = int(row[COLS.ID])
        pid = int(row[COLS.P])
        if pid != ROOT_ID:
            if np.all(row[COLS.X: COLS.R] == raw_data.get_row(pid)[COLS.X: COLS.R]):
                bad_segments.append((pid, idx))
    return len(bad_segments) == 0, bad_segments


if __name__ == '__main__':

    files = ['test_data/swc/Neuron.swc',
             'test_data/swc/Single_apical_no_soma.swc',
             'test_data/swc/Single_apical.swc',
             'test_data/swc/Single_basal.swc',
             'test_data/swc/Single_axon.swc',
             'test_data/swc/Neuron_zero_radius.swc',
             'test_data/swc/sequential_trunk_off_1_16pt.swc',
             'test_data/swc/non_sequential_trunk_off_1_16pt.swc']

    for f in files:
        rd = load_data(f)
        print 'Check file %s...' % f
        print 'Has soma? %s' % has_soma(rd)
        fr = finite_radius_neurites(rd)
        print 'All neurites have finite radius? %s' % fr[0]
        if not fr[0]:
            print 'Points with zero radius detected:', fr[1]
        ci = has_sequential_ids(rd)
        print 'Consecutive indices? %s' % ci[0]
        if not ci[0]:
            print'Non consecutive IDs detected:', ci[1]
        else:
            fs = has_finite_length_segments(rd)
            print 'Finite length segments? %s' % fs[0]
            if not fs[0]:
                print 'Segments with zero length detected:', fs[1]
