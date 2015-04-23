'''Generic tree class and iteration functions'''
from itertools import chain, imap


class Tree(object):
    '''
    Simple tree class. This is a recursive data structure, with each tree
    holding a value and a list of children trees. Every node is a tree.
    '''
    def __init__(self, value):
        self.value = value
        self.children = list()

    def add_child(self, tree):
        '''Add a child to the list of this tree's children'''
        self.children.append(tree)


def iter_preorder(tree):
    '''Depth-first pre-order iteration of tree nodes'''
    yield tree.value
    for v in chain(*imap(iter_preorder, tree.children)):
        yield v


def iter_postorder(tree):
    '''Depth-first post-order iteration of tree nodes'''
    for v in chain(*imap(iter_postorder, tree.children)):
        yield v
    yield tree.value


