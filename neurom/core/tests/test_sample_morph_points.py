import os
import numpy as np
import neurom as nm

_path = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(_path, "../../../test_data/")

def test_sample_morph_points():
    morph_path = os.path.join(TEST_DATA_PATH, 'sample_morph_points.asc')
    morph = nm.load_neuron(morph_path)
    expected_basal = np.array([
        [0., 11., 0., ],
        [0., 21., 0., ],
        [-5.65685445, 28.65685445, 0., ],
        [-13.8053271, 34.31732977, 0., ],
        [0., 31., 0., ],
        [0., 41., 0., ],
        [1.38485692, 30.30600158, 0., ],
        [11., 0., 0., ]])
    assert np.allclose(
        expected_basal,
        nm.sample_morph_points(morph, nm.NeuriteType.basal_dendrite, 10))
    expected_axon = np.array([[0., -11., 0., ]])
    assert np.allclose(
        expected_axon,
        nm.sample_morph_points(morph, nm.NeuriteType.axon, 10))
