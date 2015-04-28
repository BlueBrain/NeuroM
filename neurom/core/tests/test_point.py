from nose import tools as nt
from neurom.core.point import Point
from neurom.core.point import point_from_row

def test_point_members():
    p = Point(1, 2, 3, 4, 'FOO')
    nt.ok_(p.t == 'FOO')
    nt.ok_(p.x == 1)
    nt.ok_(p.y == 2)
    nt.ok_(p.z == 3)
    nt.ok_(p.r == 4)


def test_point_from_row():
    p = point_from_row([1, 2, 3, 4, 5, 6, 7])
    nt.ok_(p.t == 2)
    nt.ok_(p.x == 3)
    nt.ok_(p.y == 4)
    nt.ok_(p.z == 5)
    nt.ok_(p.r == 6)
