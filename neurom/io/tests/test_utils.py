'''Test neurom.io.utils'''
import os
from neurom.io.readers import load_data
from neurom.io import utils
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')

FILES = [os.path.join(SWC_PATH, f)
         for f in ['Neuron.swc',
                   'Single_apical_no_soma.swc',
                   'Single_apical.swc',
                   'Single_basal.swc',
                   'Single_axon.swc',
                   'sequential_trunk_off_0_16pt.swc',
                   'sequential_trunk_off_1_16pt.swc',
                   'sequential_trunk_off_42_16pt.swc',
                   'Neuron_no_missing_ids_no_zero_segs.swc']]

SOMA_IDS = [[1, 2, 3],
            [],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 9],
            [2, 10],
            [43, 51],
            [1, 2, 3]]

INIT_IDS = [[4, 215, 426, 637],
            [],
            [4],
            [4],
            [4],
            [2, 10],
            [3, 11],
            [44, 52],
            [4]]


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')

RAW_DATA = [load_data(f) for f in FILES]

def test_get_soma_ids():
    for i, d in enumerate(RAW_DATA):
        nt.ok_(utils.get_soma_ids(d) == SOMA_IDS[i])


def test_get_initial_segment_ids():
    for i, d in enumerate(RAW_DATA):
        nt.ok_(utils.get_initial_segment_ids(d) == INIT_IDS[i])
