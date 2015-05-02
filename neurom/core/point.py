'''Point classes and functions'''

from collections import namedtuple
from neurom.core.dataformat import COLS


Point = namedtuple('Point', ('x', 'y', 'z', 'r', 't'))


def as_point(row):
    '''Create a Point from a data block row'''
    return Point(row[COLS.X], row[COLS.Y], row[COLS.Z],
                 row[COLS.R], row[COLS.TYPE])
