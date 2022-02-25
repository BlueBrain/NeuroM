import warnings
import pytest
import neurom
import numpy as np
import numpy.testing as npt
from neurom import NeuriteType
from neurom.features import get
from neurom.features import _MORPHOLOGY_FEATURES, _NEURITE_FEATURES

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


def _morphology_features(mode):

    features = {
        "soma_radius": [
            {
                "expected_wout_subtrees": 0.5,
                "expected_with_subtrees": 0.5,
            }
        ],
        "soma_surface_area": [
            {
                "expected_wout_subtrees": np.pi,
                "expected_with_subtrees": np.pi,
            }
        ],
        "soma_volume": [
            {
                "expected_wout_subtrees": np.pi / 6.,
                "expected_with_subtrees": np.pi / 6.,
            }
        ],
        "number_of_sections_per_neurite": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [3, 5, 3],
                "expected_with_subtrees": [3, 2, 3, 3],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [3, 5],
                "expected_with_subtrees": [3, 2],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [3],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [3],
                "expected_with_subtrees": [3],
            }
        ],
        "max_radial_distance": [
            {
                # without subtrees AoD is considered a single tree, with [3, 3] being the furthest
                # with subtrees AoD subtrees are considered separately and the distance is calculated
                # from their respective roots. [1, 4] is the furthest point in this case
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 3.60555127546398,
                "expected_with_subtrees": 3.16227766016837,
            },
            {
                # with a global origin, AoD axon subtree [2, 4] is always furthest from soma
                "kwargs": {"neurite_type": NeuriteType.all, "origin": np.array([0., 0., 0.])},
                "expected_wout_subtrees": 4.47213595499958,
                "expected_with_subtrees": 4.47213595499958,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 3.60555127546398,  # [3, 3] - [0, 1]
                "expected_with_subtrees": 3.16227766016837,  # [1, 4] - [0, 1]
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite, "origin": np.array([0., 0., 0.])},
                "expected_wout_subtrees": 4.47213595499958,  # [2, 4] - [0, 0]
                "expected_with_subtrees": 4.12310562561766,  # [1, 4] - [0, 0]

            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 2.23606797749979,  # [3, 3] - [1, 2]
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon, "origin": np.array([0., 0., 0.])},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 4.47213595499958,  # [2, 4] - [0, 0]
            }
        ],
        "total_length_per_neurite": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [3., 6.82842712474619, 3.],
                "expected_with_subtrees": [3., 3.414213562373095, 3.414213562373095, 3],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [3., 6.82842712474619],
                "expected_with_subtrees": [3., 3.414213562373095],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [3.414213562373095],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [3.],
                "expected_with_subtrees": [3.],
            }
        ],
        "total_area_per_neurite" : [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1.884956, 4.290427, 1.884956],  # total_length * 2piR
                "expected_with_subtrees": [1.884956, 2.145214, 2.145214, 1.884956],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1.884956, 4.290427],
                "expected_with_subtrees": [1.884956, 2.145214],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2.145214],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.884956],
                "expected_with_subtrees": [1.884956],
            }
        ],
        "total_volume_per_neurite": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.09424778, 0.21452136, 0.09424778],  # total_length * piR^2
                "expected_with_subtrees": [0.09424778, 0.10726068, 0.10726068, 0.09424778],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.09424778, 0.21452136],
                "expected_with_subtrees": [0.09424778, 0.10726068],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.10726068],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.09424778],
                "expected_with_subtrees": [0.09424778],
            }
        ],
        "trunk_origin_azimuths": [  # Not applicable to distal subtrees
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [3.1415927, 0.0, 0.0],
                "expected_with_subtrees": [3.1415927, 0.0, 0.0],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [3.1415927, 0.0],
                "expected_with_subtrees": [3.1415927, 0.0],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [],
            },
        ],
        "trunk_origin_elevations": [  # Not applicable to distal subtrees
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.0, 1.5707964, -1.5707964],
                "expected_with_subtrees": [0.0, 1.5707964, -1.5707964],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.0, 1.5707964],
                "expected_with_subtrees": [0.0, 1.5707964],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [],
            },
        ],
        "trunk_vectors": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [[-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]],
                "expected_with_subtrees": [[-1., 0., 0.], [0., 1., 0.], [1., 2., 0.], [0., -1., 0.]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [[-1., 0., 0.], [0., 1., 0.]],
                "expected_with_subtrees": [[-1., 0., 0.], [0., 1., 0.]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [[1., 2., 0.]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [[0., -1., 0.]],
                "expected_with_subtrees": [[0., -1., 0.]],
            },

        ],
        "trunk_angles": [ # Not applicable to distal subtrees
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1.570796, 3.141592, 1.570796],
                "expected_with_subtrees": [1.570796, 3.141592, 1.570796],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1.5707964, 1.570796],
                "expected_with_subtrees": [1.5707964, 1.570796],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [],
            },
        ],
        "trunk_angles_from_vector": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
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
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [[np.pi / 2., - np.pi / 2, np.pi], [0., 0., 0.]],
                "expected_with_subtrees": [[np.pi / 2., - np.pi / 2, np.pi], [0., 0., 0.]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [[0.463648, -0.463648,  0.]],
            },

        ],
        "trunk_angles_inter_types": [
            {
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
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.1, 0.1, 0.1],
                "expected_with_subtrees": [0.1, 0.1, 0.1, 0.1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.1, 0.1],
                "expected_with_subtrees": [0.1, 0.1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.1],
                "expected_with_subtrees": [0.1],
            },
        ],
        "trunk_section_lengths": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1., 1.414213, 1.],
                "expected_with_subtrees": [1., 1.414213, 1.414213, 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1., 1.414213],
                "expected_with_subtrees": [1., 1.414213],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414213],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.],
                "expected_with_subtrees": [1.],
            },
        ],
        "number_of_neurites": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 3,
                "expected_with_subtrees": 4,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 2,
                "expected_with_subtrees": 2,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": 1,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 1,
                "expected_with_subtrees": 1,
            },
        ],
        "neurite_volume_density": [  # our morphology is flat :(
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [np.nan, np.nan, np.nan],
                "expected_with_subtrees": [np.nan, np.nan, np.nan, np.nan],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [np.nan, np.nan],
                "expected_with_subtrees": [np.nan, np.nan],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [np.nan],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [np.nan],
                "expected_with_subtrees": [np.nan],
            },
        ],
        "sholl_crossings": [
            {
                "kwargs": {"neurite_type": NeuriteType.all, "radii": [1.5, 3.5]},
                "expected_wout_subtrees": [3, 2],
                "expected_with_subtrees": [3, 2],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite, "radii": [1.5, 3.5]},
                "expected_wout_subtrees": [2, 2],
                "expected_with_subtrees": [2, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon, "radii": [1.5, 3.5]},
                "expected_wout_subtrees": [0, 0],
                "expected_with_subtrees": [0, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite, "radii": [1.5, 3.5]},
                "expected_wout_subtrees": [1, 0],
                "expected_with_subtrees": [1, 0],
            },
        ],
        "sholl_frequency": [
            {
                "kwargs": {"neurite_type": NeuriteType.all, "step_size": 3},
                "expected_wout_subtrees": [0, 2],
                "expected_with_subtrees": [0, 2],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite, "step_size": 3},
                "expected_wout_subtrees": [0, 2],
                "expected_with_subtrees": [0, 1],
            },
#            { see #987
#                "kwargs": {"neurite_type": NeuriteType.axon},
#                "kwargs": {"step_size": 3},
#                "expected_wout_subtrees": [0, 0],
#                "expected_with_subtrees": [0, 1],
#            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite, "step_size": 2},
                "expected_wout_subtrees": [0, 1],
                "expected_with_subtrees": [0, 1],
            },

        ],
        "total_width": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 6.0,
                "expected_with_subtrees": 6.0,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 6.0,
                "expected_with_subtrees": 4.0,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 2.0,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 1.0,
                "expected_with_subtrees": 1.0,
            },
        ],
        "total_height": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 7.0,
                "expected_with_subtrees": 7.0,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 4.0,
                "expected_with_subtrees": 4.0,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 2.0,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 2.0,
                "expected_with_subtrees": 2.0,
            },
        ],
        "total_depth": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 0.0,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 0.0,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 0.0,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 0.0,
            },
        ],
    }

    features_not_tested = set(_MORPHOLOGY_FEATURES) - set(features.keys())

    assert not features_not_tested, (
        "The following morphology tests need to be included in the mixed morphology tests:\n"
        f"{features_not_tested}"
    )

    for feature_name, configurations in features.items():
        for cfg in configurations:
            kwargs = cfg["kwargs"] if "kwargs" in cfg else {}

            if mode == "with-subtrees":
                expected = cfg["expected_with_subtrees"]
            elif mode == "wout-subtrees":
                expected = cfg["expected_wout_subtrees"]
            else:
                raise ValueError("Uknown mode")

            yield feature_name, kwargs, expected


@pytest.mark.parametrize("feature_name, kwargs, expected", _morphology_features(mode="wout-subtrees"))
def test_morphology__morphology_features_wout_subtrees(feature_name, kwargs, expected, mixed_morph):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        kwargs["use_subtrees"] = False

        npt.assert_allclose(
            get(feature_name, mixed_morph, **kwargs),
            expected,
            rtol=1e-6
        )


@pytest.mark.parametrize("feature_name, kwargs, expected", _morphology_features(mode="with-subtrees"))
def test_morphology__morphology_features_with_subtrees(feature_name, kwargs, expected, mixed_morph):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        kwargs["use_subtrees"] = True

        npt.assert_allclose(
            get(feature_name, mixed_morph, **kwargs),
            expected,
            rtol=1e-6
        )

"""
def _neurite_features():

    features = {}

    features_not_tested = set(_NEURITE_FEATURES) - set(features.keys())

    #assert not features_not_tested, (
    #    "The following morphology tests need to be included in the mixed morphology tests:\n"
    #    f"{features_not_tested}"
    #)

    for feature_name, configurations in features.items():
        for cfg in configurations:
            kwargs = cfg["kwargs"] if "kwargs" in cfg else {}
            yield feature_name, kwargs, cfg["expected_wout_subtrees"], cfg["expected_with_subtrees"]



@pytest.mark.parametrize("feature_name, kwargs, expected_wout_subtrees, expected_with_subtrees", _neurite_features())
def test_morphology__morphology_features(feature_name, kwargs, expected_wout_subtrees, expected_with_subtrees, mixed_morph):

    kwargs["use_subtrees"] = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

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
