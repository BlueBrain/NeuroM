'''Build sections from a tree

Sections are defined as points between forking points,
between the root node and forking points, or between
forking points and end-points

'''

from itertools import ifilter
from neurom.core import tree

REF_TREE = tree.Tree(0)
REF_TREE.add_child(tree.Tree(11))
REF_TREE.add_child(tree.Tree(12))
REF_TREE.children[0].add_child(tree.Tree(111))
REF_TREE.children[0].add_child(tree.Tree(112))
REF_TREE.children[1].add_child(tree.Tree(121))
REF_TREE.children[1].add_child(tree.Tree(122))
REF_TREE.children[1].children[0].add_child(tree.Tree(1211))
REF_TREE.children[1].children[0].children[0].add_child(tree.Tree(12111))
REF_TREE.children[1].children[0].children[0].add_child(tree.Tree(12112))
REF_TREE.children[0].children[0].add_child(tree.Tree(1111))
REF_TREE.children[0].children[0].children[0].add_child(tree.Tree(11111))
REF_TREE.children[0].children[0].children[0].add_child(tree.Tree(11112))


def is_forking_point(t):
    '''Is this tree a forking point?'''
    return len(t.children) > 1


def is_leaf(t):
    '''Is this tree a leaf?'''
    return len(t.children) == 0


def is_root(t):
    '''Is this tree the root node?'''
    return t.parent is None


def get_section(t):
    '''get the upstream section starting from this tree'''
    ui = tree.iter_upstream(t)
    sec = [ui.next()]
    for i in ui:
        sec.append(i)
        if is_forking_point(i) or is_root(i):
            break
    sec.reverse()
    return sec


def iter_leaf_or_forking(t):
    '''Iterator to all leaves or forking points of a tree'''
    return ifilter(lambda t: len(t.children) == 0 or len(t.children) > 1,
                   tree.iter_preorder(t))


def get_sections(t):
    '''get a list of sections for tree t

    This builds the sections from scratch at each call.
    '''
    nodes = [n for n in iter_leaf_or_forking(t) if not is_root(n)]
    return tuple(get_section(n) for n in nodes)


def iter_section(t):
    '''Iterator to sections of a tree.

    Resolves to a list of sub-trees forming a section.
    '''
    return iter(get_sections(t))


if __name__ == '__main__':

    for s in iter_section(REF_TREE):
        print [tt.value for tt in s]
