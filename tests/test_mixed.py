import pytest
import neurom
from neurom import NeuriteType
from neurom.features import get


@pytest.fixture
def mixed_morph():
    """
    basal_dendrite: homogeneous
    axon_on_basal_dendrite: heterogeneous
    apical_dendrite: homogeneous
    """
    return neurom.load_morphology(
    """
    1  1    0  0  0    0.5 -1
    2  3   -1  0  0    0.1  1
    3  3   -2  0  0    0.1  2
    4  3    0 -3  0    0.1  3
    5  3   -2  1  0    0.1  3
    6  3    0  1  0    0.1  1
    7  3    1  2  0    0.1  6
    8  3    1  4  0    0.1  7
    9  2    2  3  0    0.1  7
    10 2    2  4  0    0.1  9
    11 2    3  3  0    0.1  9
    12 4    0 -1  0    0.1  1
    13 4    0 -2  0    0.1 12
    14 4    0 -3  0    0.1 13
    15 4    1 -2  0    0.1 13
    """,
    reader="swc")


def test_morph_number_of_sections_per_neurite(mixed_morph):

    assert get("number_of_sections_per_neurite", mixed_morph, use_subtrees=False) == [3, 5, 3]
    assert get("number_of_sections_per_neurite", mixed_morph, use_subtrees=True) == [3, 2, 3, 3]

    assert get("number_of_sections_per_neurite", mixed_morph, neurite_type=NeuriteType.basal_dendrite, use_subtrees=False) == [3, 5]
    assert get("number_of_sections_per_neurite", mixed_morph, neurite_type=NeuriteType.basal_dendrite, use_subtrees=True) == [3, 2]

    assert get("number_of_sections_per_neurite", mixed_morph, neurite_type=NeuriteType.axon, use_subtrees=False) == []
    assert get("number_of_sections_per_neurite", mixed_morph, neurite_type=NeuriteType.axon, use_subtrees=True) == [3]

    assert get("number_of_sections_per_neurite", mixed_morph, neurite_type=NeuriteType.apical_dendrite, use_subtrees=False) == [3]
    assert get("number_of_sections_per_neurite", mixed_morph, neurite_type=NeuriteType.apical_dendrite, use_subtrees=True) == [3]

"""
def test_mixed__segment_lengths(mixed_morph):

    axon_on_dendrite = mixed_morph.neurites[0]

    get("segment_lengths", process_inhomogeneous_subtrees=True, neurite_type=NeuriteType.axon)

def test_features(mixed_morph):

    # the traditional way processes each tree as a whole
    assert get("number_of_neurites", mixed_morph, process_inhomogeneous_subtrees=False) == 1
    assert get("number_of_neurites", mixed_morph, process_inhomogeneous_subtrees=False, neurite_type=NeuriteType.basal_dendrite) == 1
    assert get("number_of_neurites", mixed_morph, process_inhomogeneous_subtrees=False, neurite_type=NeuriteType.axon) == 0

    # the new way checks for inhomogeneous subtrees anc counts them as separate neurites
    assert get("number_of_neurites", mixed_morph, process_inhomogeneous_subtrees=True) == 2
    assert get("number_of_neurites", mixed_morph, process_inhomogeneous_subtrees=True, neurite_type=NeuriteType.basal_dendrite) == 1
    assert get("number_of_neurites", mixed_morph, process_inhomogeneous_subtrees=True, neurite_type=NeuriteType.axon) == 1




def test_mixed_types(mixed_morph):

    from neurom import NeuriteType
    from neurom.features import get

    types = [neurite.type for neurite in mixed_morph.neurites]

    res = get("number_of_sections", mixed_morph, neurite_type=NeuriteType.axon)

    print(types, res)
    assert False
"""
