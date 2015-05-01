'''Neuron building example.

An example of how to build an object representing a neuron from an SWC file
'''
from itertools import imap
from neurom.io.readers import load_data
from neurom.io.utils import make_tree
from neurom.io.utils import make_neuron
from neurom.io.utils import get_soma_ids
from neurom.io.utils import get_initial_segment_ids
from neurom.core import tree
from neurom.core import neuron
from neurom.core.dataformat import COLS
from neurom.core.point import point_from_row


def point_iter(iterator):
    '''Transform tree iterator into a point iterator

    Args:
        iterator: tree iterator for a tree holding raw data rows.
    '''
    return imap(point_from_row, tree.val_iter(iterator))


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

    nrn = neuron.Neuron(soma_pts, trees)

    print 'Neuron soma raw data', [r for r in nrn.soma.iter()]
    print 'Neuron soma points', [point_from_row(p)
                                 for p in nrn.soma.iter()]

    print 'Neuron tree init points, types'
    for tt in nrn.neurite_trees:
        print tt.value[COLS.ID], tt.value[COLS.TYPE]

    print 'Making neuron 2'
    nrn2 = make_neuron(rd)
    print 'Neuron 2 soma points', [r for r in nrn2.soma.iter()]
    print 'Neuron 2 soma points', [point_from_row(p)
                                   for p in nrn2.soma.iter()]
    print 'Neuron 2 tree init points, types'
    for tt in nrn2.neurite_trees:
        print tt.value[COLS.ID], tt.value[COLS.TYPE]

    print 'Print neuron leaves as points'
    for tt in nrn2.neurite_trees:
        for p in point_iter(tree.iter_leaf(tt)):
            print p
