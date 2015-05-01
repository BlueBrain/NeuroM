'''Tree morphometrics functions '''
from neurom.core.tree import iter_segment
from neurom.analysis.morphmath import point_dist
from neurom.core.point import point_from_row
from neurom.core.point import Point


def get_segment_lengths(tree):
    ''' return a list of segments length inside tree
    '''
    return [point_dist(point_from_row(s[0]), point_from_row(s[1])) for s in iter_segment(tree)]


def get_segment_diameters(tree):
    ''' return a list of segments diameter inside tree
    '''
    return [point_from_row(s[0]).r + point_from_row(s[1]).r for s in iter_segment(tree)]


def get_segment_radialdists(position, tree):
    ''' return a list of radial distance of segment to given point'''
    pos = point_from_row(position)
    return [point_dist(pos, Point((point_from_row(s[0]).x + point_from_row(s[1]).x) / 2.0,
                                  (point_from_row(s[0]).y + point_from_row(s[1]).y) / 2.0,
                                  (point_from_row(s[0]).z + point_from_row(s[1]).z) / 2.0,
                                  0.0, 0.0)) for s in iter_segment(tree)]
