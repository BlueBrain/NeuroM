from neurom.io.readers import load_data
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE


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


if __name__ == '__main__':

    files = ['test_data/swc/Neuron.swc',
             'test_data/swc/Single_apical_no_soma.swc',
             'test_data/swc/Single_apical.swc',
             'test_data/swc/Single_basal.swc',
             'test_data/swc/Single_axon.swc',
             'test_data/swc/Neuron_zero_radius.swc']

    for f in files:
        rd = load_data(f)
        print 'Check file %s...' % f
        print 'Has soma? %s' % has_soma(rd)
        fr = finite_radius_neurites(rd)
        print 'All neurites have finite radius? %s' % fr[0]
        if not fr[0]:
            print 'Points with zero radius detected:', fr[1]
