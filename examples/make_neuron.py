'''Neuron building example.

An example of how to build an object representing a neuron from an SWC file
'''
from neurom.io.readers import load_data
from neurom.io.utils import make_tree
from neurom.io.utils import get_soma_ids
from neurom.io.utils import get_initial_segment_ids
from neurom.core import tree
from neurom.core.dataformat import COLS
from neurom.core.point import point_from_row


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


class SomaA(object):
    '''Type A soma'''
    def __init__(self, points):
        _point = point_from_row(points[0])
        self.points = points
        self.center = _point[:3]
        self.radius = _point.r


class SomaB(object):
    '''Type B soma'''
    def __init__(self, points):
        self.points = points
        self.center = None
        self.radius = None


class SomaC(object):
    '''Type C soma'''
    def __init__(self, points):
        self.points = points
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
        for p in tree.val_iter(tree.iter_preorder(tr)):
            print point_from_row(p)

    print 'Initial segment IDs:', init_seg_ids

    nrn = neuron(soma_pts, trees)

    print 'Neuron soma points', nrn.soma.points
    print 'Neuron tree init points, types'
    for tt in nrn.neurite_trees:
        print tt.value[COLS.ID], tt.value[COLS.TYPE]

    print 'Making neuron 2'
    nrn2 = make_neuron(rd)
    print 'Neuron 2 soma points', nrn2.soma.points
    print 'Neuron 2 tree init points, types'
    for tt in nrn2.neurite_trees:
        print tt.value[COLS.ID], tt.value[COLS.TYPE]
