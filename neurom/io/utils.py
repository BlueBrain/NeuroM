'''Utility functions and classes for higher level RawDataWrapper access'''
import itertools
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE


def get_soma_ids(rdw):
    '''Returns a list of IDs of points that are somas'''
    return rdw.get_ids(lambda r: r[COLS.TYPE] == POINT_TYPE.SOMA)


def get_initial_segment_ids(rdw):
    '''Returns a list of IDs of initial tree segments

    These are defined as non-soma points whose perent is a soma point.
    '''
    l = list(itertools.chain(*[rdw.get_children(s) for s in get_soma_ids(rdw)]))
    return [i for i in l if rdw.get_row(i)[COLS.TYPE] != POINT_TYPE.SOMA]
