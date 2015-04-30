'''Neuron building example.

An example of how to build an object representing a neuron from an SWC file
'''
from itertools import imap
from neurom.io.readers import load_data
from neurom.io.utils import make_tree
from neurom.io.utils import get_soma_ids
from neurom.io.utils import get_initial_segment_ids
from neurom.core import tree
from neurom.core.dataformat import COLS
from neurom.core.point import point_from_row


def point_iter(iterator):
    '''Transform tree iterator into a point iterator

    Args:
        iterator: tree iterator for a tree holding raw data rows.
    '''
    return imap(point_from_row, tree.val_iter(iterator))


class SOMA_TYPE(object):
    '''Enumeration holding soma types

    Type A: single point at centre
    Type B: Three points on circumference of sphere
    Type C: More than three points
    INVALID: Not satisfying any of the above
    '''
    INVALID, A, B, C = xrange(4)

    @staticmethod
    def get_type(points):
        '''gues what this does?'''
        npoints = len(points)
        return {1: SOMA_TYPE.A,
                3: SOMA_TYPE.B,
                2: SOMA_TYPE.INVALID}.get(npoints, SOMA_TYPE.C)


class BaseSoma(object):
    '''Base class for a soma.

    Holds a list of raw data rows corresponding to soma points
    and provides iterator access to them.
    '''
    def __init__(self, points):
        self._points = points

    def iter(self):
        '''Iterator to soma contents'''
        return iter(self._points)

    def iter_point(self):
        '''Point iterator so soma contents

        Returns neurom.core.point.Point objects.
        '''
        return imap(point_from_row, self._points)


class SomaA(BaseSoma):
    '''Type A soma'''
    def __init__(self, points):
        super(SomaA, self).__init__(points)
        _point = point_from_row(points[0])
        self.center = _point[:3]
        self.radius = _point.r


class SomaB(BaseSoma):
    '''Type B soma'''
    def __init__(self, points):
        super(SomaB, self).__init__(points)
        self.center = None
        self.radius = None


class SomaC(BaseSoma):
    '''Type C soma'''
    def __init__(self, points):
        super(SomaC, self).__init__(points)
        self.center = None
        self.radius = None


def make_soma(points):
    '''toy soma'''
    stype = SOMA_TYPE.get_type(points)
    return {SOMA_TYPE.A: SomaA,
            SOMA_TYPE.B: SomaB,
            SOMA_TYPE.C: SomaC}[stype](points)


class neuron(object):
    '''Toy neuron class for testing ideas'''
    def __init__(self, soma_points, neurite_trees):
        self.soma = make_soma(soma_points)
        self.neurite_trees = neurite_trees


def make_neuron(raw_data):
    '''Build a neuron from a raw data block'''
    _trees = [make_tree(raw_data, iseg)
              for iseg in get_initial_segment_ids(raw_data)]
    _soma_pts = [rd.get_row(s_id) for s_id in get_soma_ids(raw_data)]
    return neuron(_soma_pts, _trees)


if __name__ == '__main__':

    filename = 'test_data/swc/Neuron.swc'

    rd = load_data(filename)

    init_seg_ids = get_initial_segment_ids(rd)

    trees = [make_tree(rd, sg) for sg in init_seg_ids]

    soma_pts = [rd.get_row(si) for si in get_soma_ids(rd)]

    for tr in trees:
        for p in point_iter(tree.iter_preorder(tr)):
            print p

    print 'Initial segment IDs:', init_seg_ids

    nrn = neuron(soma_pts, trees)

    print 'Neuron soma raw data', [r for r in nrn.soma.iter()]
    print 'Neuron soma points', [p for p in nrn.soma.iter_point()]

    print 'Neuron tree init points, types'
    for tt in nrn.neurite_trees:
        print tt.value[COLS.ID], tt.value[COLS.TYPE]

    print 'Making neuron 2'
    nrn2 = make_neuron(rd)
    print 'Neuron 2 soma points', [r for r in nrn2.soma.iter()]
    print 'Neuron 2 soma points', [p for p in nrn2.soma.iter_point()]
    print 'Neuron 2 tree init points, types'
    for tt in nrn2.neurite_trees:
        print tt.value[COLS.ID], tt.value[COLS.TYPE]

    print 'Print neuron leaves as points'
    for tt in nrn2.neurite_trees:
        for p in point_iter(tree.iter_leaf(tt)):
            print p
