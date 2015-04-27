import os
import numpy as np
from neurom.io.readers import load_data
from neurom.io import check
from neurom.core.dataformat import ROOT_ID
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')


def test_has_sequential_ids_good_data():

    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Single_apical_no_soma.swc',
                       'Single_apical.swc',
                       'Single_basal.swc',
                       'Single_axon.swc',
                       'Neuron_zero_radius.swc',
                       'sequential_trunk_off_0_16pt.swc',
                       'sequential_trunk_off_1_16pt.swc',
                       'sequential_trunk_off_42_16pt.swc',
                       'Neuron_no_missing_ids_no_zero_segs.swc']
             ]

    for f in files:
        ok, ids = check.has_sequential_ids(load_data(f))
        nt.ok_(ok)
        nt.ok_(len(ids) == 0)



def test_has_sequential_ids_bad_data():

    f = os.path.join(SWC_PATH, 'Neuron_missing_ids.swc')

    ok, ids = check.has_sequential_ids(load_data(f))
    nt.ok_(not ok)
    nt.ok_(ids == [6, 217, 428, 639])


def test_has_soma_good_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Single_apical.swc',
                       'Single_basal.swc',
                       'Single_axon.swc']]

    for f in files:
        nt.ok_(check.has_soma(load_data(f)))


def test_has_soma_bad_data():
    f = os.path.join(SWC_PATH, 'Single_apical_no_soma.swc')
    nt.ok_(not check.has_soma(load_data(f)))


def test_has_finite_radius_neurites_good_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Single_apical.swc',
                       'Single_basal.swc',
                       'Single_axon.swc']]

    for f in files:
        ok, ids = check.has_all_finite_radius_neurites(load_data(f))
        nt.ok_(ok)
        nt.ok_(len(ids) == 0)


def test_has_finite_radius_neurites_bad_data():
    f = os.path.join(SWC_PATH, 'Neuron_zero_radius.swc')
    ok, ids = check.has_all_finite_radius_neurites(load_data(f))
    nt.ok_(not ok)
    nt.ok_(ids == [194, 210, 246, 304, 493])


def test_has_finite_length_segments_good_data():
    files = [os.path.join(SWC_PATH, f)
             for f in [
                       'sequential_trunk_off_0_16pt.swc',
                       'sequential_trunk_off_1_16pt.swc',
                       'sequential_trunk_off_42_16pt.swc']]
    for f in files:
        ok, ids = check.has_all_finite_length_segments(load_data(f))
        nt.ok_(ok)
        nt.ok_(len(ids) == 0)


def test_has_finite_length_segments_bad_data():
    files = [os.path.join(SWC_PATH, f)
             for f in ['Neuron.swc',
                       'Single_apical.swc',
                       'Single_basal.swc',
                       'Single_axon.swc']]

    bad_segs = [[(4, 5), (215, 216),
                 (426, 427), (637, 638)],
                [(4, 5)],
                [(4, 5)],
                [(4, 5)]]

    for i, f in enumerate(files):
        ok, ids = check.has_all_finite_length_segments(load_data(f))
        nt.ok_(not ok)
        nt.assert_equal(ids, bad_segs[i])
