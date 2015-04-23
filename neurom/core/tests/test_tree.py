from nose import tools as nt
from neurom.core.Tree import Tree
from neurom.core.Tree import iter_preorder
from neurom.core.Tree import iter_postorder
from neurom.core.Tree import iter_upstream


REF_TREE = Tree(0)
REF_TREE.add_child(Tree(11))
REF_TREE.add_child(Tree(12))
REF_TREE.children[0].add_child(Tree(111))
REF_TREE.children[0].add_child(Tree(112))
REF_TREE.children[1].add_child(Tree(121))
REF_TREE.children[1].add_child(Tree(122))


def test_instantiate_tree():
    t = Tree('hello')
    nt.ok_(t.parent is None)
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


def test_upstream_iteration():

    nt.ok_(list(iter_upstream(REF_TREE)) == [0])
    nt.ok_(list(iter_upstream(REF_TREE.children[0])) == [11, 0])
    nt.ok_(list(iter_upstream(REF_TREE.children[0].children[0])) ==
           [111, 11, 0])
    nt.ok_(list(iter_upstream(REF_TREE.children[0].children[1])) ==
           [112, 11, 0])


    nt.ok_(list(iter_upstream(REF_TREE.children[1])) == [12, 0])
    nt.ok_(list(iter_upstream(REF_TREE.children[1].children[0])) ==
           [121, 12, 0])
    nt.ok_(list(iter_upstream(REF_TREE.children[1].children[1])) ==
           [122, 12, 0])


def test_children():
    nt.ok_(REF_TREE.children[0].value == 11)
    nt.ok_(REF_TREE.children[1].value == 12)
    nt.ok_(REF_TREE.children[0].children[0].value == 111)
    nt.ok_(REF_TREE.children[0].children[1].value == 112)
    nt.ok_(REF_TREE.children[1].children[0].value == 121)
    nt.ok_(REF_TREE.children[1].children[1].value == 122)


def test_add_child():
    t = Tree(0)
    t.add_child(Tree(11))
    t.add_child(Tree(22))
    nt.ok_(t.value == 0)
    nt.ok_(len(t.children) == 2)
    nt.ok_([i.value for i in t.children] == [11, 22])


def test_parent():
    t = Tree(0)
    for i in xrange(10):
        t.add_child(Tree(i))

    nt.ok_(len(t.children) == 10)

    for c in t.children:
        nt.ok_(c.parent is t)
