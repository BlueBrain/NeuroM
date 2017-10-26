import os

from nose import tools as nt
import numpy as np
import subprocess

from neurom.apps.cut_plane_detection import find_cut_plane, _get_probabilities, _create_1d_distributions
from neurom.core import Neuron
from neurom import load_neuron


def test_empty_neuron():
    result = find_cut_plane(Neuron())
    nt.assert_equal(result['status'], 'Empty neuron')


def test_create_1d_distributions():

    def tridim_array(points):
        """Duplicate column for X,Y,Z axes"""
        return np.array([points] * 3).transpose()

    points = [1.5] * 10 + [1.7] + [3.7] * 20 + [5.5] + [6.5] * 30
    hist = _create_1d_distributions(tridim_array(points), bin_width=2)
    nt.assert_equal(set(hist.keys()), set(['X', 'Y', 'Z']))
    histo, bins = hist['X'][0]
    nt.assert_equal(bins.tolist(), [1.5, 3.5, 5.5, 7.5])
    nt.assert_equal(histo.tolist(), [11, 20, 31])

    histo, bins = hist['X'][1]
    nt.assert_equal(bins.tolist(), [0.5, 2.5, 4.5, 6.5])
    nt.assert_equal(histo.tolist(), [11, 20, 31])


def test_get_probabilities():
    histo_left = ([10, 20, 30], None)
    histo_right = ([1, 2, 3, 4], None)
    sample = {'X': (histo_left, histo_right)}
    probs = list(_get_probabilities(sample))
    nt.assert_equal(probs, [('X', 0, 10., histo_left),
                            ('X', -1, 4.0, histo_right)])


def test_cut_neuron():
    result = find_cut_plane(load_neuron('test_data/valid_set/Neuron_slice.h5'))
    nt.assert_equal(result['status'], 'ok')

    cut_axis, position = result['cut_plane']
    nt.assert_equal(cut_axis, 'Z')
    nt.assert_true(abs(position - 48) < 2)


def test_display():
    result = find_cut_plane(load_neuron('test_data/valid_set/Neuron_slice.h5'), display=True)
    for fig, ax in result['figures'].values():
        nt.assert_true(fig)
        nt.assert_true(ax)
    nt.assert_equal(set(result['figures'].keys()), set(['distrib_1d', 'xy', 'xz', 'yz']))


def test_repaired_neuron():
    result = find_cut_plane(load_neuron('test_data/h5/v1/bio_neuron-000.h5'))
    nt.assert_not_equal(result['status'], 'ok')


def test_apps():
    with open(os.devnull, 'w') as fnull:
        subprocess.call(
            ['./apps/cut_plane_detection', 'test_data/valid_set/Neuron_slice.h5'], stdout=fnull)
