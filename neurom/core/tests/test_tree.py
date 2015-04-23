from nose import tools as nt
from neurom.core.Tree import Tree, iter_preorder, iter_postorder


REF_TREE = Tree(0)
REF_TREE.add_child(Tree(11))
REF_TREE.add_child(Tree(12))
REF_TREE.children[0].add_child(Tree(111))
REF_TREE.children[0].add_child(Tree(112))
REF_TREE.children[1].add_child(Tree(121))
REF_TREE.children[1].add_child(Tree(122))


def test_instantiate_tree():
    t = Tree('hello')
    nt.ok_(t.value == 'hello')
    nt.ok_(len(t.children) == 0)


def test_preorder_iteration():
    nt.ok_(list(iter_preorder(REF_TREE)) == [0, 11, 111, 112, 12, 121, 122])
    nt.ok_(list(iter_preorder(REF_TREE.children[0])) == [11, 111, 112])
    nt.ok_(list(iter_preorder(REF_TREE.children[1])) == [12, 121, 122])


def test_postorder_iteration():
    nt.ok_(list(iter_postorder(REF_TREE)) == [111, 112, 11, 121, 122, 12, 0])
    nt.ok_(list(iter_postorder(REF_TREE.children[0])) == [111, 112, 11])
    nt.ok_(list(iter_postorder(REF_TREE.children[1])) == [121, 122, 12])


def test_add_child():
    t = Tree(0)
    t.add_child(Tree(11))
    t.add_child(Tree(22))
    nt.ok_(t.value == 0)
    nt.ok_(len(t.children) == 2)
    nt.ok_([i.value for i in t.children] == [11, 22])

