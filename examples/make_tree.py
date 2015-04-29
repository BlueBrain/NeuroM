'''Examples of tree building functions'''
import itertools
from neurom.io.readers import load_data
from neurom.core import tree
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE
from neurom.core.dataformat import ROOT_ID
from neurom.core.point import point_from_row


def add_children(t, rdw):
    '''Add children to a tree'''
    for c in rdw.get_children(t.value[COLS.ID]):
        child = tree.Tree(rdw.get_row(c))
        t.add_child(child)
        add_children(child, rdw)
    return t


def make_tree(rdw, root_id=ROOT_ID):
    '''Return a tree obtained from a raw data block'''
    head_node = tree.Tree(rdw.get_row(root_id))
    add_children(head_node, rdw)
    return head_node


def get_soma_ids(rdw):
    '''Returns a list of IDs of points that are somas'''
    return rdw.get_ids(lambda r: r[COLS.TYPE] == 1)


def get_initial_segment_ids(rdw):
    '''Returns a list of IDs of initial neurite segments

    These are defined as non-soma points whose perent is a soma point.
    '''
    l = list(itertools.chain(*[rdw.get_children(s) for s in get_soma_ids(rdw)]))
    return [i for i in l if rdw.get_row(i)[COLS.TYPE] != POINT_TYPE.SOMA]


class dummy_neuron(object):
    '''Tpy neuron class for testing ideas'''
    def __init__(self, soma_points, neurite_trees):
        self.soma_points = soma_points
        self.neurite_trees = neurite_trees


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

    nrn = dummy_neuron(soma_pts, trees)

    print 'Neuron soma points', nrn.soma_points
    print 'Neuron tree init points, types'
    for tt in nrn.neurite_trees:
        print tt.value[COLS.ID], tt.value[COLS.TYPE]
