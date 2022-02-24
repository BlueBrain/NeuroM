import pytest
import neurom
import numpy as np
import numpy.testing as npt
from neurom import NeuriteType
from neurom.features import get
from neurom.features import _MORPHOLOGY_FEATURES

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
        "soma_radius": [
            {
                "neurite_type": None,
                "expected_wout_subtrees": 0.5,
                "expected_with_subtrees": 0.5,
            }
        ],
        "soma_surface_area": [
            {
                "neurite_type": None,
                "expected_wout_subtrees": np.pi,
                "expected_with_subtrees": np.pi,
            }
        ],
        "soma_volume": [
            {
                "neurite_type": None,
                "expected_wout_subtrees": np.pi / 6.,
                "expected_with_subtrees": np.pi / 6.,
            }
        ],
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
        ],
        "total_area_per_neurite" : [
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": [1.884956, 4.290427, 1.884956],  # total_length * 2piR
                "expected_with_subtrees": [1.884956, 2.145214, 2.145214, 1.884956],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": [1.884956, 4.290427],
                "expected_with_subtrees": [1.884956, 2.145214],
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2.145214],
            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "expected_wout_subtrees": [1.884956],
                "expected_with_subtrees": [1.884956],
            }
        ],
        "total_volume_per_neurite": [
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": [0.09424778, 0.21452136, 0.09424778],  # total_length * piR^2
                "expected_with_subtrees": [0.09424778, 0.10726068, 0.10726068, 0.09424778],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": [0.09424778, 0.21452136],
                "expected_with_subtrees": [0.09424778, 0.10726068],
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.10726068],
            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "expected_wout_subtrees": [0.09424778],
                "expected_with_subtrees": [0.09424778],
            }
        ],
        "trunk_origin_azimuths": [  # Not applicable to distal subtrees
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": [3.1415927, 0.0, 0.0],
                "expected_with_subtrees": [3.1415927, 0.0, 0.0],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": [3.1415927, 0.0],
                "expected_with_subtrees": [3.1415927, 0.0],
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [],
            },
        ],
        "trunk_origin_elevations": [  # Not applicable to distal subtrees
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": [0.0, 1.5707964, -1.5707964],
                "expected_with_subtrees": [0.0, 1.5707964, -1.5707964],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": [0.0, 1.5707964],
                "expected_with_subtrees": [0.0, 1.5707964],
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [],
            },
        ],
        "trunk_vectors": [
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": [[-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]],
                "expected_with_subtrees": [[-1., 0., 0.], [0., 1., 0.], [1., 2., 0.], [0., -1., 0.]],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": [[-1., 0., 0.], [0., 1., 0.]],
                "expected_with_subtrees": [[-1., 0., 0.], [0., 1., 0.]],
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [[1., 2., 0.]],
            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "expected_wout_subtrees": [[0., -1., 0.]],
                "expected_with_subtrees": [[0., -1., 0.]],
            },

        ],
        "trunk_angles": [ # Not applicable to distal subtrees
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": [1.570796, 3.141592, 1.570796],
                "expected_with_subtrees": [1.570796, 3.141592, 1.570796],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": [1.5707964, 1.570796],
                "expected_with_subtrees": [1.5707964, 1.570796],
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [],
            },
        ],
        "trunk_angles_from_vector": [
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": [
                    [np.pi / 2., - np.pi / 2, np.pi],
                    [0., 0., 0.],
                    [np.pi, np.pi, 0.],
                ],
                "expected_with_subtrees": [
                    [np.pi / 2., - np.pi / 2, np.pi],
                    [0., 0., 0.],
                    [0.463648, -0.463648,  0.],
                    [np.pi, np.pi, 0.],
                ],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": [[np.pi / 2., - np.pi / 2, np.pi], [0., 0., 0.]],
                "expected_with_subtrees": [[np.pi / 2., - np.pi / 2, np.pi], [0., 0., 0.]],
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [[0.463648, -0.463648,  0.]],
            },

        ],
        "trunk_angles_inter_types": [
            {
                "neurite_type": None,
                "kwargs": {
                    "source_neurite_type": NeuriteType.basal_dendrite,
                    "target_neurite_type": NeuriteType.axon,
                },
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [
                    [[ 2.034444,  1.107149, -3.141593]],
                    [[ 0.463648, -0.463648,  0.      ]],
                ],
            },
        ],
        "trunk_origin_radii": [
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": [0.1, 0.1, 0.1],
                "expected_with_subtrees": [0.1, 0.1, 0.1, 0.1],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": [0.1, 0.1],
                "expected_with_subtrees": [0.1, 0.1],
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.1],
            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "expected_wout_subtrees": [0.1],
                "expected_with_subtrees": [0.1],
            },
        ],
        "trunk_section_lengths": [
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": [1., 1.414213, 1.],
                "expected_with_subtrees": [1., 1.414213, 1.414213, 1.],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": [1., 1.414213],
                "expected_with_subtrees": [1., 1.414213],
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414213],
            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "expected_wout_subtrees": [1.],
                "expected_with_subtrees": [1.],
            },
        ],
        "number_of_neurites": [
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": 3,
                "expected_with_subtrees": 4,
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": 2,
                "expected_with_subtrees": 2,
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": [],
                "expected_with_subtrees": 1,
            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "expected_wout_subtrees": 1,
                "expected_with_subtrees": 1,
            },
        ],
        "sholl_crossings": [
            {
                "neurite_type": NeuriteType.all,
                "kwargs": {"radii": [1.5, 3.5]},
                "expected_wout_subtrees": [3, 2],
                "expected_with_subtrees": [3, 2],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "kwargs": {"radii": [1.5, 3.5]},
                "expected_wout_subtrees": [2, 2],
                "expected_with_subtrees": [2, 1],
            },
            {
                "neurite_type": NeuriteType.axon,
                "kwargs": {"radii": [1.5, 3.5]},
                "expected_wout_subtrees": [0, 0],
                "expected_with_subtrees": [0, 1],
            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "kwargs": {"radii": [1.5, 3.5]},
                "expected_wout_subtrees": [1, 0],
                "expected_with_subtrees": [1, 0],
            },
        ],
        "sholl_frequency": [
            {
                "neurite_type": NeuriteType.all,
                "kwargs": {"step_size": 3},
                "expected_wout_subtrees": [0, 2],
                "expected_with_subtrees": [0, 2],
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "kwargs": {"step_size": 3},
                "expected_wout_subtrees": [0, 2],
                "expected_with_subtrees": [0, 1],
            },
#            { see #987
#                "neurite_type": NeuriteType.axon,
#                "kwargs": {"step_size": 3},
#                "expected_wout_subtrees": [0, 0],
#                "expected_with_subtrees": [0, 1],
#            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "kwargs": {"step_size": 2},
                "expected_wout_subtrees": [0, 1],
                "expected_with_subtrees": [0, 1],
            },

        ],
        "total_width": [
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": 6.0,
                "expected_with_subtrees": 6.0,
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": 6.0,
                "expected_with_subtrees": 4.0,
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 2.0,
            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "expected_wout_subtrees": 1.0,
                "expected_with_subtrees": 1.0,
            },
        ],
        "total_height": [
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": 7.0,
                "expected_with_subtrees": 7.0,
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": 4.0,
                "expected_with_subtrees": 4.0,
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 2.0,
            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "expected_wout_subtrees": 2.0,
                "expected_with_subtrees": 2.0,
            },
        ],
        "total_depth": [
            {
                "neurite_type": NeuriteType.all,
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 0.0,
            },
            {
                "neurite_type": NeuriteType.basal_dendrite,
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 0.0,
            },
            {
                "neurite_type": NeuriteType.axon,
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 0.0,
            },
            {
                "neurite_type": NeuriteType.apical_dendrite,
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 0.0,
            },
        ],
    }

    #features_not_tested = set(_MORPHOLOGY_FEATURES) - set(features.keys())

    #assert not features_not_tested, (
    #    "The following morphology tests need to be included in the mixed morphology tests:\n"
    #    f"{features_not_tested}"
    #)

    for feature_name, configurations in features.items():
        for cfg in configurations:
            kwargs = cfg["kwargs"] if "kwargs" in cfg else {}
            yield feature_name, cfg["neurite_type"], kwargs, cfg["expected_wout_subtrees"], cfg["expected_with_subtrees"]


@pytest.mark.parametrize("feature_name, neurite_type, kwargs, expected_wout_subtrees, expected_with_subtrees", _morphology_features())
def test_features__morphology(feature_name, neurite_type, kwargs, expected_wout_subtrees, expected_with_subtrees, mixed_morph):

    kwargs["use_subtrees"] = False

    if neurite_type is not None:
        kwargs["neurite_type"] = neurite_type

    npt.assert_allclose(
        get(feature_name, mixed_morph, **kwargs),
        expected_wout_subtrees,
        rtol=1e-6
    )

    kwargs["use_subtrees"] = True
    npt.assert_allclose(
        get(feature_name, mixed_morph, **kwargs),
        expected_with_subtrees,
        rtol=1e-6
    )

"""
def test_mixed_types(mixed_morph):

    from neurom import NeuriteType
    from neurom.features import get

    types = [neurite.type for neurite in mixed_morph.neurites]

    res = get("number_of_sections", mixed_morph, neurite_type=NeuriteType.axon)

    print(types, res)
    assert False
"""
