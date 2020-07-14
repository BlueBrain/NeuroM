import os
import numpy as np
import neurom as nm

_path = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(_path, "../../../test_data/")


def test_iter_positions():
    morph_path = os.path.join(TEST_DATA_PATH, 'iter_positions.asc')
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
    basal_filter = lambda s: s.type == nm.NeuriteType.basal_dendrite
    assert np.allclose(
        expected_basal,
        np.array(list(
            nm.iter_positions(morph, basal_filter, 10))))
    expected_axon = np.array([[0., -11., 0., ]])
    axon_filter = lambda s: s.type == nm.NeuriteType.axon
    assert np.allclose(
        expected_axon,
        np.array(list(
            nm.iter_positions(morph, axon_filter, 10))))


def test_iter_positions_no_skips_branch_points():
    morph = nm.load_neuron(os.path.join(TEST_DATA_PATH, 'swc', 'simple.swc'))
    points = nm.iter_positions(morph, None, 1)
    exp_points = np.array([
        [0, 1, 0],
        [0, 2, 0],
        [0, 3, 0],
        [0, 4, 0],
        [0, 5, 0],
        [-1, 5, 0],
        [-2, 5, 0],
        [-3, 5, 0],
        [-4, 5, 0],
        [-5, 5, 0],
        [1, 5, 0],
        [2, 5, 0],
        [3, 5, 0],
        [4, 5, 0],
        [5, 5, 0],
        [6, 5, 0],
        [0, -1, 0],
        [0, -2, 0],
        [0, -3, 0],
        [0, -4, 0],
        [1, -4, 0],
        [2, -4, 0],
        [3, -4, 0],
        [4, -4, 0],
        [5, -4, 0],
        [6, -4, 0],
        [-1, -4, 0],
        [-2, -4, 0],
        [-3, -4, 0],
        [-4, -4, 0],
        [-5, -4, 0]
    ])

    np.testing.assert_almost_equal(exp_points, np.array(list(points)))
