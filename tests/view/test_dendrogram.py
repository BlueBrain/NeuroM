from pathlib import Path
from collections import namedtuple

import numpy as np
import neurom.view.dendrogram as dm
from neurom import load_morphology, get
from neurom.core.types import NeuriteType

from numpy.testing import assert_array_almost_equal

DATA_PATH = Path(__file__).parent.parent / 'data'
NEURON_PATH = DATA_PATH / 'h5/v1/Neuron.h5'


def test_create_dendrogram_morphology():
    m = load_morphology(NEURON_PATH)
    dendrogram = dm.Dendrogram(m)
    assert NeuriteType.soma == dendrogram.neurite_type
    soma_len = 1.0
    assert soma_len == dendrogram.height
    assert soma_len == dendrogram.width
    assert_array_almost_equal(
        [[-0.5, 0], [-0.5, soma_len], [0.5, soma_len], [0.5, 0]], dendrogram.coords
    )
    assert len(m.neurites) == len(dendrogram.children)


def test_dendrogram_get_coords():
    segment_lengts = np.array([0, 1, 1])
    segment_radii = np.array([0.5, 1, 0.25])
    coords = dm.Dendrogram.get_coords(segment_lengts, segment_radii)
    assert_array_almost_equal([[-0.5, 0], [-1, 1], [-0.25, 2], [0.25, 2], [1, 1], [0.5, 0]], coords)


def test_create_dendrogram_neurite():
    def assert_trees(neurom_section, dendrogram):
        assert len(neurom_section.children) == len(dendrogram.children)
        for i, d in enumerate(dendrogram.children):
            section = neurom_section.children[i]
            assert section.type == d.neurite_type

    m = load_morphology(NEURON_PATH)
    neurite = m.neurites[0]
    dendrogram = dm.Dendrogram(neurite)
    assert neurite.type == dendrogram.neurite_type
    assert_trees(neurite.root_node, dendrogram)


def test_move_positions():
    origin = [10, -10]
    positions = {1: [0, 0], 2: [3, -3]}
    moved_positions = dm.move_positions(positions, origin)
    assert list(positions.keys()) == list(moved_positions.keys())
    assert [10, -10] == moved_positions[1].tolist()
    assert [13, -13] == moved_positions[2].tolist()


def test_get_size():
    DendrogramMock = namedtuple('Dendrogram', 'height')
    dendrogram1 = DendrogramMock(10)
    dendrogram2 = DendrogramMock(1)
    positions = {dendrogram1: [-1, 0], dendrogram2: [3, 3]}
    w, h = dm.get_size(positions)
    assert 4 == w
    assert 10 == h


def test_layout_dendrogram():
    def assert_layout(dendrogram):
        for i, child in enumerate(dendrogram.children):
            # child is higher than parent in Y coordinate
            assert positions[child][1] >= positions[dendrogram][1] + dendrogram.height
            if i < len(dendrogram.children) - 1:
                next_child = dendrogram.children[i + 1]
                # X space between child is enough for their widths
                assert positions[next_child][0] - positions[child][0] > 0.5 * (
                    next_child.width + child.width
                )
            assert_layout(child)

    m = load_morphology(NEURON_PATH)
    dendrogram = dm.Dendrogram(m)
    positions = dm.layout_dendrogram(dendrogram, np.array([0, 0]))
    assert_layout(dendrogram)


def test_morphology_not_corrupted():
    # Regression for #492: dendrogram was corrupting morphology used to construct it.
    # This caused the section path distance calculation to raise a KeyError exception.
    m = load_morphology(NEURON_PATH)
    dm.Dendrogram(m)
    assert len(get('section_path_distances', m)) > 0
