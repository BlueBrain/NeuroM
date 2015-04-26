import os
import numpy as np
from neurom.io import readers
from neurom.core.dataformat import COLS
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')


def check_single_section_random_swc(data, fmt, offset=0):
    nt.ok_(fmt == 'SWC')
    nt.ok_(offset == 0)
    nt.ok_(len(data) == 16)
    nt.ok_(np.shape(data) == (16, 7))


def test_read_swc_basic():
    data, offset, fmt = readers.read_swc(
        os.path.join(SWC_PATH,
                     'random_trunk_off_0_16pt.swc'))

    check_single_section_random_swc(data, fmt, offset)

class TestRawDataWrapper_SingleSectionRandom(object):
    def setup(self):
        self.data = readers.load_data(
            os.path.join(SWC_PATH, 'sequential_trunk_off_42_16pt.swc'))
        self.first_id = int(self.data.data_block[0][COLS.ID])

    def test_data_structure(self):
        check_single_section_random_swc(self.data.data_block,
                                        self.data.fmt)

    def test_get_ids(self):
        nt.ok_(self.data.get_ids() == range(self.first_id, self.first_id+16))


    @nt.raises(LookupError)
    def test_get_parent_invalid_id_raises(self):
        self.data.get_parent(-1)
        self.data.get_parent(-2)
        self.data.get_parent(16)

    @nt.raises(LookupError)
    def test_get_children_invalid_id_raises(self):
        self.data.get_children(-2)

    def test_fork_points_is_empty(self):
        nt.ok_(len(self.data.get_fork_points()) == 0)

    def test_get_parent(self):
        for i, idx in enumerate(self.data.get_ids()):
            if i == 0:
                nt.assert_equal(self.data.get_parent(idx), -1)
            else:
                nt.assert_equal(self.data.get_parent(idx), idx - 1)

    def test_get_children(self):
        ids = self.data.get_ids()
        last = ids[-1]
        for i in self.data.get_ids():
            children = self.data.get_children(i)
            if i != last:
                nt.ok_(children == [i + 1])
            else:
                nt.ok_(len(children) == 0)

        nt.ok_(self.data.get_children(readers.ROOT_ID) == [ids[0]])

    def test_get_endpoints(self):
        # end-point is last point
        nt.ok_(self.data.get_end_points() == [self.data.data_block[-1][COLS.ID]])


    def test_get_row(self):
        for i in self.data.get_ids():
            r = self.data.get_row(i)
            ii = i - self.first_id
            nt.ok_(len(r) == 7)
            nt.ok_(r[1] >= 0 and r[1] < 8)
            nt.ok_(r[1] == ii % 8)
            nt.ok_(r[2] == ii)
            nt.ok_(r[3] == ii)
            nt.ok_(r[4] == ii)
            nt.ok_(r[5] == ii)

    def test_get_point(self):
        for i in self.data.get_ids():
            p = self.data.get_point(i)
            ii = i - self.first_id
            nt.ok_(p)
            nt.ok_(p.t >= 0 and p.t < 8)
            nt.ok_(p.t == ii % 8)
            nt.ok_(p.x == ii)
            nt.ok_(p.y == ii)
            nt.ok_(p.z == ii)
            nt.ok_(p.r == ii)

    def test_iter_row(self):
        for i, p in enumerate(self.data.iter_row()):
            ii = i + self.first_id
            pid = -1 if i == 0 else ii - 1
            nt.assert_true(np.all(p == (ii, i%8, i, i, i, i, pid)))

    @nt.raises(LookupError)
    def test_iter_row_low_id_raises(self):
        self.data.iter_row(-1)

    @nt.raises(LookupError)
    def test_iter_row_high_id_raises(self):
        self.data.iter_row(16 + self.first_id)
