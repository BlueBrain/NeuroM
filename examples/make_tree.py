from neurom.io.readers import load_data
from neurom.core import tree
from neurom.core.dataformat import Rows
from neurom.core.point import point_from_row


def add_children(t, rdw):
    '''Add children to a tree'''
    for c in rdw.get_children(t.value[Rows.ID]):
        child = tree.Tree(rdw.get_row(c))
        t.add_child(child)
        add_children(child, rdw)
    return t


def make_trees(rdw):
    '''Return a list of trees obtained from a raw data block'''
    head_nodes = [tree.Tree(r) for r in rdw.iter_row() if r[Rows.P] == -1]
    for h in head_nodes:
        add_children(h, rdw)
    return head_nodes


if __name__ == '__main__':

    filename = 'test_data/swc/Neuron.swc'

    rd = load_data(filename)

    trees = make_trees(rd)

    for p in tree.val_iter(tree.iter_preorder(trees[0])):
        print point_from_row(p)
