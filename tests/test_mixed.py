import warnings
import pytest
import neurom
import numpy as np
import numpy.testing as npt
from neurom import NeuriteType
from neurom.features import get
from neurom.features import _MORPHOLOGY_FEATURES, _NEURITE_FEATURES
import collections.abc

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

def _assert_feature_equal(obj, feature_name, expected_values, kwargs, use_subtrees):

    def innermost_value(iterable):
        while isinstance(iterable, collections.abc.Iterable):
            try:
                iterable = iterable[0]
            except IndexError:
                # empty list
                return None
        return iterable


    assert_equal = lambda a, b: npt.assert_equal(
        a, b, err_msg=f"ACTUAL: {a}\nDESIRED: {b}", verbose=False
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        values = get(feature_name, obj, use_subtrees=use_subtrees, **kwargs)

        # handle empty lists because allclose always passes in that case.
        # See: https://github.com/numpy/numpy/issues/11071
        if isinstance(values, collections.abc.Iterable):
            if isinstance(expected_values, collections.abc.Iterable):
                if isinstance(innermost_value(values), (float, np.floating)):
                    npt.assert_allclose(values, expected_values, atol=1e-5)
                else:
                    assert_equal(values, expected_values)
            else:
                assert_equal(values, expected_values)
        else:
            if isinstance(expected_values, collections.abc.Iterable):
                assert_equal(values, expected_values)
            else:
                if isinstance(values, (float, np.floating)):
                    npt.assert_allclose(values, expected_values, atol=1e-5)
                else:
                    assert_equal(values, expected_values)


def _dispatch_features(features, mode):

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
                "expected_wout_subtrees": 0,
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

    return _dispatch_features(features, mode)


@pytest.mark.parametrize("feature_name, kwargs, expected", _morphology_features(mode="wout-subtrees"))
def test_morphology__morphology_features_wout_subtrees(feature_name, kwargs, expected, mixed_morph):
    _assert_feature_equal(mixed_morph, feature_name, expected, kwargs, use_subtrees=False)


@pytest.mark.parametrize("feature_name, kwargs, expected", _morphology_features(mode="with-subtrees"))
def test_morphology__morphology_features_with_subtrees(
    feature_name, kwargs, expected, mixed_morph
):
    _assert_feature_equal(mixed_morph, feature_name, expected, kwargs, use_subtrees=True)


def _neurite_features(mode):

    features = {
        "number_of_segments": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 11,
                "expected_with_subtrees": 11,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 8,
                "expected_with_subtrees": 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0,
                "expected_with_subtrees": 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 3,
                "expected_with_subtrees": 3,
            },
        ],
        "number_of_leaves": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 7,
                "expected_with_subtrees": 7,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 5,
                "expected_with_subtrees": 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0,
                "expected_with_subtrees": 2,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 2,
                "expected_with_subtrees": 2,
            },
        ],
        "total_length": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 12.828427,
                "expected_with_subtrees": 12.828427,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 9.828427,
                "expected_with_subtrees": 6.414214,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0.,
                "expected_with_subtrees": 3.414214,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 3.,
                "expected_with_subtrees": 3.,
            }
        ],
        "total_area": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 8.060339,
                "expected_with_subtrees": 8.060339,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 6.175383,
                "expected_with_subtrees": 4.030170,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0.,
                "expected_with_subtrees": 2.145214,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 1.884956,
                "expected_with_subtrees": 1.884956,
            }
        ],
        "total_volume": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 0.403016,
                "expected_with_subtrees": 0.403016,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 0.308769,
                "expected_with_subtrees": 0.201508,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0.,
                "expected_with_subtrees": 0.107261,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 0.0942478,
                "expected_with_subtrees": 0.0942478,
            }
        ],
        "section_lengths": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1., 1., 1., 1.414214, 2., 1.414214, 1., 1., 1., 1., 1.],
                "expected_with_subtrees":
                    [1., 1., 1., 1.414214, 2., 1.414214, 1., 1., 1., 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [1., 1., 1., 1.414214, 2., 1.414214, 1., 1.],
                "expected_with_subtrees":
                    [1., 1., 1., 1.414214, 2.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414214, 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1., 1., 1.],
                "expected_with_subtrees": [1., 1., 1.],
            }
        ],
        "section_areas": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [0.628318, 0.628319, 0.628319, 0.888577, 1.256637, 0.888576,
                     0.628318, 0.628318, 0.628318, 0.628318, 0.628318],
                "expected_with_subtrees":
                    [0.628318, 0.628319, 0.628319, 0.888577, 1.256637, 0.888576,
                     0.628318, 0.628318, 0.628318, 0.628318, 0.628318],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [0.628318, 0.628319, 0.628319, 0.888577, 1.256637, 0.888576,
                     0.628318, 0.628318],
                "expected_with_subtrees":
                    [0.628318, 0.628319, 0.628319, 0.888577, 1.256637],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.888576, 0.628318, 0.628318],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.628318, 0.628318, 0.628318],
                "expected_with_subtrees": [0.628318, 0.628318, 0.628318],
            }

        ],
        "section_volumes": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [0.031415, 0.031415, 0.031415, 0.044428, 0.062831, 0.044428, 0.031415,
                     0.031415, 0.031415, 0.031415, 0.031415],
                "expected_with_subtrees":
                    [0.031415, 0.031415, 0.031415, 0.044428, 0.062831, 0.044428, 0.031415,
                     0.031415, 0.031415, 0.031415, 0.031415],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [0.031415, 0.031415, 0.031415, 0.044428, 0.062831, 0.044428, 0.031415,
                     0.031415],
                "expected_with_subtrees":
                    [0.031415, 0.031415, 0.031415, 0.044428, 0.062831],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.044428, 0.031415, 0.031415],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.031415, 0.031415, 0.031415],
                "expected_with_subtrees": [0.031415, 0.031415, 0.031415],
            }
        ],
        "section_tortuosity": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1.0] * 11,
                "expected_with_subtrees": [1.0] * 11,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1.0] * 8,
                "expected_with_subtrees": [1.0] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.0] * 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.0] * 3,
                "expected_with_subtrees": [1.0] * 3,
            }
        ],
        "section_radial_distances": [
            {
                # radial distances change when the mixed subtrees are processed because
                # the root of the subtree is considered
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1.0, 2.0, 1.4142135, 1.4142135, 3.1622777, 2.828427,
                     3.6055512, 3.6055512, 1.0, 2.0, 1.4142135],
                "expected_with_subtrees":
                    [1.0, 2.0, 1.4142135, 1.4142135, 3.1622777, 1.414214,
                     2.236068, 2.236068, 1.0, 2.0, 1.4142135],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [1.0, 2.0, 1.4142135, 1.4142135, 3.1622777, 2.828427,
                     3.6055512, 3.6055512],
                "expected_with_subtrees":
                    [1.0, 2.0, 1.4142135, 1.4142135, 3.1622777],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414214, 2.236068, 2.236068],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1., 2., 1.414214],
                "expected_with_subtrees": [1., 2., 1.414214],
            }

        ],
        "section_term_radial_distances": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [2.0, 1.4142135, 3.1622777, 3.6055512, 3.6055512, 2.0, 1.4142135],
                "expected_with_subtrees":
                    [2.0, 1.4142135, 3.1622777, 2.236068, 2.236068, 2.0, 1.4142135],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [2.0, 1.4142135, 3.1622777, 3.6055512, 3.6055512],
                "expected_with_subtrees": [2.0, 1.4142135, 3.1622777],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2.236068, 2.236068],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [2., 1.414214],
                "expected_with_subtrees": [2., 1.414214],
            }

        ],
        "section_bif_radial_distances": [
            {
                # radial distances change when the mixed subtrees are processed because
                # the root of the subtree is considered instead of the tree root
                # heterogeneous forks are not valid forking points
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1.0, 1.4142135, 2.828427, 1.0],
                "expected_with_subtrees": [1.0, 1.4142135, 1.0],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1.0, 1.4142135, 2.828427],
                "expected_with_subtrees": [1.0],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.4142135,],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.],
                "expected_with_subtrees": [1.],
            }
        ],
        "section_end_distances": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1., 1., 1., 1.414214, 2., 1.414214, 1., 1., 1., 1., 1.],
                "expected_with_subtrees":
                    [1., 1., 1., 1.414214, 2., 1.414214, 1., 1., 1., 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [1., 1., 1., 1.414214, 2., 1.414214, 1., 1.],
                "expected_with_subtrees":
                    [1., 1., 1., 1.414214, 2.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414214, 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1., 1., 1.],
                "expected_with_subtrees": [1., 1., 1.],
            }
        ],
        "section_term_lengths": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1., 1., 2., 1., 1., 1., 1.],
                "expected_with_subtrees": [1., 1., 2., 1., 1., 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1., 1., 2., 1., 1.],
                "expected_with_subtrees": [1., 1., 2.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1., 1.],
                "expected_with_subtrees": [1., 1.],
            }
        ],
        "section_taper_rates": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.0] * 11,
                "expected_with_subtrees": [0.0] * 11,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.0] * 8,
                "expected_with_subtrees": [0.0] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.0] * 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.0] * 3,
                "expected_with_subtrees": [0.0] * 3,
            }
        ],
        "section_bif_lengths": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1., 1.414214, 1.414214, 1.],
                "expected_with_subtrees": [1., 1.414214, 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1., 1.414214, 1.414214],
                "expected_with_subtrees": [1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414214],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.],
                "expected_with_subtrees": [1.],
            },
        ],
        "section_branch_orders": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0, 1, 1, 0, 1, 1, 2, 2, 0, 1, 1],
                "expected_with_subtrees": [0, 1, 1, 0, 1, 1, 2, 2, 0, 1, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0, 1, 1, 0, 1, 1, 2, 2],
                "expected_with_subtrees": [0, 1, 1, 0, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1, 2, 2],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0, 1, 1],
                "expected_with_subtrees": [0, 1, 1],
            },
        ],
        "section_bif_branch_orders": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0, 0, 1, 0],
                "expected_with_subtrees": [0, 1, 0],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0, 0, 1],
                "expected_with_subtrees": [0],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0],
                "expected_with_subtrees": [0],
            },
        ],
        "section_term_branch_orders": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1, 1, 1, 2, 2, 1, 1],
                "expected_with_subtrees": [1, 1, 1, 2, 2, 1, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1, 1, 1, 2, 2],
                "expected_with_subtrees": [1, 1, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2, 2],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1, 1],
                "expected_with_subtrees": [1, 1],
            },
        ],
        "section_strahler_orders": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1],
                "expected_with_subtrees": [2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [2, 1, 1, 2, 1, 2, 1, 1],
                "expected_with_subtrees": [2, 1, 1, 2, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2, 1, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [2, 1, 1],
                "expected_with_subtrees": [2, 1, 1],
            },
        ],
        "segment_lengths": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1., 1., 1., 1.414214, 2., 1.414214, 1., 1., 1., 1., 1.],
                "expected_with_subtrees":
                    [1., 1., 1., 1.414214, 2., 1.414214, 1., 1., 1., 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [1., 1., 1., 1.414214, 2., 1.414214, 1., 1.],
                "expected_with_subtrees":
                    [1., 1., 1., 1.414214, 2.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414214, 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1., 1., 1.],
                "expected_with_subtrees": [1., 1., 1.],
            }
        ],
        "segment_areas": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [0.628318, 0.628319, 0.628319, 0.888577, 1.256637, 0.888576,
                     0.628318, 0.628318, 0.628318, 0.628318, 0.628318],
                "expected_with_subtrees":
                    [0.628318, 0.628319, 0.628319, 0.888577, 1.256637, 0.888576,
                     0.628318, 0.628318, 0.628318, 0.628318, 0.628318],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [0.628318, 0.628319, 0.628319, 0.888577, 1.256637, 0.888576,
                     0.628318, 0.628318],
                "expected_with_subtrees":
                    [0.628318, 0.628319, 0.628319, 0.888577, 1.256637],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.888576, 0.628318, 0.628318],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.628318, 0.628318, 0.628318],
                "expected_with_subtrees": [0.628318, 0.628318, 0.628318],
            }
        ],
        "segment_volumes": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [0.031415, 0.031415, 0.031415, 0.044428, 0.062831, 0.044428, 0.031415,
                     0.031415, 0.031415, 0.031415, 0.031415],
                "expected_with_subtrees":
                    [0.031415, 0.031415, 0.031415, 0.044428, 0.062831, 0.044428, 0.031415,
                     0.031415, 0.031415, 0.031415, 0.031415],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [0.031415, 0.031415, 0.031415, 0.044428, 0.062831, 0.044428, 0.031415,
                     0.031415],
                "expected_with_subtrees":
                    [0.031415, 0.031415, 0.031415, 0.044428, 0.062831],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.044428, 0.031415, 0.031415],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.031415, 0.031415, 0.031415],
                "expected_with_subtrees": [0.031415, 0.031415, 0.031415],
            }
        ],
        "segment_radii": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.1] * 11,
                "expected_with_subtrees": [0.1] * 11,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.1] * 8,
                "expected_with_subtrees": [0.1] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.1] * 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.1] * 3,
                "expected_with_subtrees": [0.1] * 3,
            }
        ],
        "segment_taper_rates": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.0] * 11,
                "expected_with_subtrees": [0.0] * 11,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.0] * 8,
                "expected_with_subtrees": [0.0] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.0] * 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.0] * 3,
                "expected_with_subtrees": [0.0] * 3,
            },
        ],
        "segment_radial_distances": [
            {
                # radial distances change when the mixed subtrees are processed because
                # the root of the subtree is considered
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [0.5, 1.5, 1.118034, 0.70710677, 2.236068, 2.1213202, 3.2015622, 3.2015622,
                     0.5, 1.5, 1.118034],
                "expected_with_subtrees":
                    [0.5, 1.5, 1.118034, 0.70710677, 2.236068,  0.707107, 1.802776, 1.802776,
                     0.5, 1.5, 1.118034],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [0.5, 1.5, 1.118034, 0.70710677, 2.236068, 2.1213202, 3.2015622, 3.2015622],
                "expected_with_subtrees":
                    [0.5, 1.5, 1.118034, 0.70710677, 2.236068],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.707107, 1.802776, 1.802776],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.5, 1.5, 1.118034],
                "expected_with_subtrees": [0.5, 1.5, 1.118034],
            },
        ],
        "segment_midpoints": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [
                    [-1.5, 0.0, 0.0], [-2.5, 0.0, 0.0], [-2.0, 0.5, 0.0], [0.5, 1.5, 0. ],
                    [1.0, 3.0, 0.0], [1.5, 2.5, 0.0], [2.0, 3.5, 0.0], [2.5, 3.0, 0.0],
                    [0.0, -1.5, 0.0], [0., -2.5, 0.0], [0.5, -2.0, 0.0]],
                "expected_with_subtrees": [
                    [-1.5, 0.0, 0.0], [-2.5, 0.0, 0.0], [-2.0, 0.5, 0.0], [0.5, 1.5, 0. ],
                    [1.0, 3.0, 0.0], [1.5, 2.5, 0.0], [2.0, 3.5, 0.0], [2.5, 3.0, 0.0],
                    [0.0, -1.5, 0.0], [0., -2.5, 0.0], [0.5, -2.0, 0.0]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [
                    [-1.5, 0.0, 0.0], [-2.5, 0.0, 0.0], [-2.0, 0.5, 0.0], [0.5, 1.5, 0. ],
                    [1.0, 3.0, 0.0], [1.5, 2.5, 0.0], [2.0, 3.5, 0.0], [2.5, 3.0, 0.0]],
                "expected_with_subtrees": [
                    [-1.5, 0.0, 0.0], [-2.5, 0.0, 0.0], [-2.0, 0.5, 0.0], [0.5, 1.5, 0. ],
                    [1.0, 3.0, 0.0]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [[1.5, 2.5, 0.0], [2.0, 3.5, 0.0], [2.5, 3.0, 0.0]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [[0.0, -1.5, 0.0], [0., -2.5, 0.0], [0.5, -2.0, 0.0]],
                "expected_with_subtrees": [[0.0, -1.5, 0.0], [0., -2.5, 0.0], [0.5, -2.0, 0.0]],
            },
        ],
        "segment_meander_angles": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [],
            },
        ],
        "number_of_sections": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 11,
                "expected_with_subtrees": 11,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 8,
                "expected_with_subtrees": 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0,
                "expected_with_subtrees": 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 3,
                "expected_with_subtrees": 3,
            },
        ],
        "number_of_bifurcations": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 4,
                "expected_with_subtrees": 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 3,
                "expected_with_subtrees": 1,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0,
                "expected_with_subtrees": 1,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 1,
                "expected_with_subtrees": 1,
            },
        ],
        "number_of_forking_points": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 4,
                "expected_with_subtrees": 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 3,
                "expected_with_subtrees": 1,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0,
                "expected_with_subtrees": 1,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 1,
                "expected_with_subtrees": 1,
            },
        ],
        "volume_density": [ # neurites are flat :(
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": np.nan,
                "expected_with_subtrees": np.nan,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": np.nan,
                "expected_with_subtrees": np.nan,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0,
                "expected_with_subtrees": np.nan,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": np.nan,
                "expected_with_subtrees": np.nan,
            },
        ],
        "local_bifurcation_angles": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.5 * np.pi, 0.785398, 0.5 * np.pi, 0.5 * np.pi],
                "expected_with_subtrees": [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.5 * np.pi, 0.785398, 0.5 * np.pi],
                "expected_with_subtrees": [1.570796],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.5 * np.pi],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.5 * np.pi],
                "expected_with_subtrees": [0.5 * np.pi],
            },
        ],
        "remote_bifurcation_angles": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.5 * np.pi, 0.785398, 0.5 * np.pi, 0.5 * np.pi],
                "expected_with_subtrees": [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.5 * np.pi, 0.785398, 0.5 * np.pi],
                "expected_with_subtrees": [1.570796],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.5 * np.pi],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.5 * np.pi],
                "expected_with_subtrees": [0.5 * np.pi],
            },
        ],
        "sibling_ratios": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1.0] * 4,
                "expected_with_subtrees": [1.0] * 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1.0] * 3,
                "expected_with_subtrees": [1.0],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.0],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.0],
                "expected_with_subtrees": [1.0],
            },
        ],
        "partition_pairs": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [(1.0, 1.0), (1.0, 3.0), (1.0, 1.0), (1.0, 1.0)],
                "expected_with_subtrees": [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [(1.0, 1.0), (1.0, 3.0), (1.0, 1.0)],
                "expected_with_subtrees": [(1.0, 1.0)],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [(1.0, 1.0)],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [(1.0, 1.0)],
                "expected_with_subtrees": [(1.0, 1.0)],
            },
        ],
        "diameter_power_relations": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [2.0] * 4,
                "expected_with_subtrees": [2.0] * 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [2.0] * 3,
                "expected_with_subtrees": [2.0],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2.0],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [2.0],
                "expected_with_subtrees": [2.0],
            },
        ],
        "bifurcation_partitions": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1.0, 3.0, 1.0, 1.0],
                "expected_with_subtrees": [1.0] * 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1.0, 3.0, 1.0],
                "expected_with_subtrees": [1.0],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.0],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.0],
                "expected_with_subtrees": [1.0],
            },
        ],
    }

    features_not_tested = list(
        set(_NEURITE_FEATURES) - set(features.keys()) - set(_MORPHOLOGY_FEATURES)
    )

#    assert not features_not_tested, (
#        "The following morphology tests need to be included in the tests:\n\n" +
#        "\n".join(sorted(features_not_tested)) + "\n"
#    )

    return _dispatch_features(features, mode)


@pytest.mark.parametrize(
    "feature_name, kwargs, expected", _neurite_features(mode="wout-subtrees")
)
def test_morphology__neurite_features_wout_subtrees(feature_name, kwargs, expected, mixed_morph):
    _assert_feature_equal(mixed_morph, feature_name, expected, kwargs, use_subtrees=False)


@pytest.mark.parametrize(
    "feature_name, kwargs, expected", _neurite_features(mode="with-subtrees")
)
def test_morphology__neurite_features_with_subtrees(feature_name, kwargs, expected, mixed_morph):
    _assert_feature_equal(mixed_morph, feature_name, expected, kwargs, use_subtrees=True)
