import pytest
import neurom
import numpy as np
import numpy.testing as npt
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
    4  3   -3  0  0    0.1  3
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


def _morphology_features():

    features = {
        "number_of_sections_per_neurite": [
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": [3, 5, 3],
                "expected_with_subtrees": [3, 2, 3, 3],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": [3, 5],
                "expected_with_subtrees": [3, 2],
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [3],
            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "expected_wout_subtrees": [3],
                "expected_with_subtrees": [3],
            }
        ],
        "max_radial_distance": [
            {
                # without subtrees AoD is considered a single tree, with [3, 3] being the furthest
                # with subtrees AoD subtrees are considered separately and the distance is calculated
                # from their respective roots. [1, 4] is the furthest point in this case
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": 3.60555127546398,
                "expected_with_subtrees": 3.16227766016837,
            },
            {
                # with a global origin, AoD axon subtree [2, 4] is always furthest from soma
                "neurite_type": NeuriteType.all,
                "kwargs": {"origin": np.array([0., 0., 0.])},
                "expected_wout_subtrees": 4.47213595499958,
                "expected_with_subtrees": 4.47213595499958,
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": 3.60555127546398,  # [3, 3] - [0, 1]
                "expected_with_subtrees": 3.16227766016837,  # [1, 4] - [0, 1]
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "kwargs": {"origin": np.array([0., 0., 0.])},
                "expected_wout_subtrees": 4.47213595499958,  # [2, 4] - [0, 0]
                "expected_with_subtrees": 4.12310562561766,  # [1, 4] - [0, 0]

            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 2.23606797749979,  # [3, 3] - [1, 2]
            },
            {
                "neurite_type": NeuriteType.axon,
                "kwargs": {"origin": np.array([0., 0., 0.])},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 4.47213595499958,  # [2, 4] - [0, 0]
            }
        ],
        "total_length_per_neurite": [
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": [3., 6.82842712474619, 3.],
                "expected_with_subtrees": [3., 3.414213562373095, 3.414213562373095, 3],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": [3., 6.82842712474619],
                "expected_with_subtrees": [3., 3.414213562373095],
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [3.414213562373095],
            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "expected_wout_subtrees": [3.],
                "expected_with_subtrees": [3.],
            }
        ]
    }

    # TODO: Add check here to ensure that there are no features not addressed

    for feature_name, configurations in features.items():
        for cfg in configurations:
            kwargs = cfg["kwargs"] if "kwargs" in cfg else {}
            yield feature_name, cfg["neurite_type"], kwargs, cfg["expected_wout_subtrees"], cfg["expected_with_subtrees"]


@pytest.mark.parametrize("feature_name, neurite_type, kwargs, expected_wout_subtrees, expected_with_subtrees", _morphology_features())
def test_features__morphology(feature_name, neurite_type, kwargs, expected_wout_subtrees, expected_with_subtrees, mixed_morph):

    npt.assert_allclose(get(feature_name, mixed_morph, neurite_type=neurite_type, use_subtrees=False, **kwargs), expected_wout_subtrees)
    npt.assert_allclose(get(feature_name, mixed_morph, neurite_type=neurite_type, use_subtrees=True, **kwargs), expected_with_subtrees)

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
