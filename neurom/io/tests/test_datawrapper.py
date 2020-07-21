"""Test neurom.io.utils."""
from pathlib import Path
import numpy as np
from nose import tools as nt

from neurom.io import datawrapper as dw
from neurom.core.dataformat import POINT_TYPE, ROOT_ID


def test__merge_sections():
    default_sec = dw.DataBlockSection()

    sec_a = dw.DataBlockSection([], ntype=0, pid=-1)
    sec_b = dw.DataBlockSection([], ntype=0, pid=-1)
    dw._merge_sections(sec_a, sec_b)
    nt.eq_(sec_a, default_sec)

    sec_a = dw.DataBlockSection(range(10), ntype=1, pid=1)
    sec_b = dw.DataBlockSection(range(9, 20), ntype=10, pid=10)
    dw._merge_sections(sec_a, sec_b)
    nt.eq_(sec_a, default_sec)
    nt.eq_(sec_b.ids, list(range(20))) # Note: 9 is in this list from sec_a, not from sec_b
    nt.eq_(sec_b.ntype, 1)
    nt.eq_(sec_b.pid, 1)


#def test__section_end_points():
#    _section_end_points
#
#def test__extract_sections():
#    pass

#DataWrapper
#neurite_root_section_ids
#soma_points

def test_DataBlockSection_str():
    s = str(dw.DataBlockSection())
    nt.ok_('DataBlockSection' in s)

def test_BlockNeuronBuilder():
    builder = dw.BlockNeuronBuilder()
    soma_points = np.array([[0, 0, 0, 1]])
    builder.add_section(0, ROOT_ID, POINT_TYPE.SOMA, soma_points)
    wrapped = builder.get_datawrapper()
    nt.eq_(len(wrapped.data_block), 1)

    builder.add_section(1, 0, POINT_TYPE.AXON, np.array([[1, 0, 0, 1]]))
    wrapped = builder.get_datawrapper()
    nt.eq_(len(wrapped.data_block), 2)

    #add child before parent, also, don't have contiguous section numbers
    # meaning that the ids will be renumbered
    builder.add_section(10, 2, POINT_TYPE.APICAL_DENDRITE, np.array([[10, 0, 0, 1]]))
    builder.add_section(2, 0, POINT_TYPE.APICAL_DENDRITE, np.array([[2, 0, 0, 1]]))
    wrapped = builder.get_datawrapper()
    nt.eq_(len(wrapped.data_block), 4)
    np.testing.assert_allclose(
        wrapped.data_block,
        np.array([[ 0., 0., 0., 1., 1., 0., -1.],
                  [ 1., 0., 0., 1., 2., 1.,  0.],
                  [ 2., 0., 0., 1., 4., 2.,  0.],
                  [10., 0., 0., 1., 4., 3.,  2.]]))
