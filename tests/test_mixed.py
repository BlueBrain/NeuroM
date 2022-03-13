import sys
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
    1  1   0  0  0   0.5 -1
    2  3  -1  0  0   0.1  1
    3  3  -2  0  0   0.1  2
    4  3  -3  0  0   0.1  3
    5  3  -3  0  1   0.1  4
    6  3  -3  0 -1   0.1  4
    7  3  -2  1  0   0.1  3
    8  3   0  1  0   0.1  1
    9  3   1  2  0   0.1  8
    10 3   1  4  0   0.1  9
    11 3   1  4  1   0.1 10
    12 3   1  4 -1   0.1 10
    13 2   2  3  0   0.1  9
    14 2   2  4  0   0.1 13
    15 2   3  3  0   0.1 13
    16 2   3  3  1   0.1 15
    17 2   3  3 -1   0.1 15
    18 4   0 -1  0   0.1  1
    19 4   0 -2  0   0.1 18
    20 4   0 -3  0   0.1 19
    21 4   0 -3  1   0.1 20
    22 4   0 -3 -1   0.1 20
    23 4   1 -2  0   0.1 19
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
                "expected_wout_subtrees": [5, 9, 5],
                "expected_with_subtrees": [5, 4, 5, 5],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [5, 9],
                "expected_with_subtrees": [5, 4],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [5],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [5],
                "expected_with_subtrees": [5],
            }
        ],
        "max_radial_distance": [
            {
                # without subtrees AoD is considered a single tree, with [3, 3] being the furthest
                # with subtrees AoD subtrees are considered separately and the distance is calculated
                # from their respective roots. [1, 4] is the furthest point in this case
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 3.741657,
                "expected_with_subtrees": 3.316625,
            },
            {
                # with a global origin, AoD axon subtree [2, 4] is always furthest from soma
                "kwargs": {"neurite_type": NeuriteType.all, "origin": np.array([0., 0., 0.])},
                "expected_wout_subtrees": 4.47213595499958,
                "expected_with_subtrees": 4.47213595499958,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 3.741657,
                "expected_with_subtrees": 3.316625,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite, "origin": np.array([0., 0., 0.])},
                "expected_wout_subtrees": 4.472136,
                "expected_with_subtrees": 4.242641,

            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 2.44949,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon, "origin": np.array([0., 0., 0.])},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 4.47213595499958,
            }
        ],
        "total_length_per_neurite": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [5., 10.828427, 5.],
                "expected_with_subtrees": [5., 5.414213, 5.414213, 5.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [5., 10.828427],
                "expected_with_subtrees": [5., 5.414214],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [5.414214],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [5.],
                "expected_with_subtrees": [5.],
            }
        ],
        "total_area_per_neurite" : [
            {
                # total length x 2piR
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [3.141593, 6.803702, 3.141593],
                "expected_with_subtrees": [3.141593, 3.401851, 3.401851, 3.141593],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [3.141593, 6.803702],
                "expected_with_subtrees": [3.141593, 3.401851],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [3.401851],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [3.141593],
                "expected_with_subtrees": [3.141593],
            }
        ],
        "total_volume_per_neurite": [
            # total_length * piR^2
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.15708 , 0.340185, 0.15708 ],
                "expected_with_subtrees": [0.15708 , 0.170093, 0.170093, 0.15708],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.15708 , 0.340185],
                "expected_with_subtrees": [0.15708 , 0.170093],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.170093],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.15708],
                "expected_with_subtrees": [0.15708],
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
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.0],
                "expected_with_subtrees": [0.0],
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
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [-1.570796],
                "expected_with_subtrees": [-1.570796],
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
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.],
                "expected_with_subtrees": [0.],
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
        "neurite_volume_density": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.235619, 0.063785, 0.235619],
                "expected_with_subtrees": [0.235619, 0.255139, 0.170093, 0.235619],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.235619, 0.063785],
                "expected_with_subtrees": [0.235619, 0.255139],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.170093],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.235619],
                "expected_with_subtrees": [0.235619],
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
            {
                "kwargs": {"neurite_type": NeuriteType.axon, "step_size": 3},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0, 1],
            },
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
                "expected_wout_subtrees": 2.0,
                "expected_with_subtrees": 2.0,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 2.0,
                "expected_with_subtrees": 2.0,
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
        "volume_density": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 0.01570426,
                "expected_with_subtrees": 0.01570426,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 0.02983588,
                "expected_with_subtrees": 0.04907583,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": np.nan,
                "expected_with_subtrees": 0.17009254,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 0.23561945,
                "expected_with_subtrees": 0.23561945,
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
                "expected_wout_subtrees": 19,
                "expected_with_subtrees": 19,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 14,
                "expected_with_subtrees": 9,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0,
                "expected_with_subtrees": 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 5,
                "expected_with_subtrees": 5,
            },
        ],
        "number_of_leaves": [
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
        "total_length": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 20.828427,
                "expected_with_subtrees": 20.828427,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 15.828427,
                "expected_with_subtrees": 10.414214,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 5.414214,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 5.,
                "expected_with_subtrees": 5.,
            }
        ],
        "total_area": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 13.086887,
                "expected_with_subtrees": 13.086887,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 9.945294,
                "expected_with_subtrees": 6.543443,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0.,
                "expected_with_subtrees": 3.401851,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 3.141593,
                "expected_with_subtrees": 3.141593,
            }
        ],
        "total_volume": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 0.654344,
                "expected_with_subtrees": 0.654344,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 0.497265,
                "expected_with_subtrees": 0.327172,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0.,
                "expected_with_subtrees": 0.170093,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 0.15708,
                "expected_with_subtrees": 0.15708,
            }
        ],
        "section_lengths": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1.] * 5 + [1.414214, 2., 1., 1.] + [1.414214, 1., 1., 1., 1.] + [1.] * 5,
                "expected_with_subtrees":
                    [1.] * 5 + [1.414214, 2., 1., 1.] + [1.414214, 1., 1., 1., 1.] + [1.] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [1.] * 5 + [1.414214, 2., 1., 1.] + [1.414214, 1., 1., 1., 1],
                "expected_with_subtrees":
                    [1.] * 5 + [1.414214, 2., 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414214, 1., 1., 1., 1.],

            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.] * 5,
                "expected_with_subtrees": [1.] * 5,
            }
        ],
        "section_areas": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [0.628318] * 5 + [0.888577, 1.256637, 0.628319, 0.628319] +
                    [0.888577, 0.628319, 0.628319, 0.628319, 0.628319] + [0.628318] * 5,
                "expected_with_subtrees":
                    [0.628318] * 5 + [0.888577, 1.256637, 0.628319, 0.628319] +
                    [0.888577, 0.628319, 0.628319, 0.628319, 0.628319] + [0.628318] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [0.628318] * 5 + [0.888577, 1.256637, 0.628319, 0.628319] +
                    [0.888577, 0.628319, 0.628319, 0.628319, 0.628319],
                "expected_with_subtrees":
                    [0.628318] * 5 + [0.888577, 1.256637, 0.628319, 0.628319],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.888577, 0.628319, 0.628319, 0.628319, 0.628319],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.628318] * 5,
                "expected_with_subtrees": [0.628318] * 5,
            }

        ],
        "section_volumes": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [0.031416] * 5 + [0.044429, 0.062832, 0.031416, 0.031416] +
                    [0.044429, 0.031416, 0.031416, 0.031416, 0.031416] +
                    [0.031416] * 5,
                "expected_with_subtrees":
                    [0.031416] * 5 + [0.044429, 0.062832, 0.031416, 0.031416] +
                    [0.044429, 0.031416, 0.031416, 0.031416, 0.031416] +
                    [0.031416] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [0.031416] * 5 + [0.044429, 0.062832, 0.031416, 0.031416] +
                    [0.044429, 0.031416, 0.031416, 0.031416, 0.031416],
                "expected_with_subtrees":
                    [0.031416] * 5 + [0.044429, 0.062832, 0.031416, 0.031416],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.044429, 0.031416, 0.031416, 0.031416, 0.031416],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.031415] * 5,
                "expected_with_subtrees": [0.031415] * 5,
            }
        ],
        "section_tortuosity": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1.0] * 19,
                "expected_with_subtrees": [1.0] * 19,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1.0] * 14,
                "expected_with_subtrees": [1.0] * 9,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.0] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.0] * 5,
                "expected_with_subtrees": [1.0] * 5,
            }
        ],
        "section_radial_distances": [
            {
                # radial distances change when the mixed subtrees are processed because
                # the root of the subtree is considered
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1.      , 2.      , 2.236068, 2.236068, 1.414214] +
                    [1.414214, 3.162278, 3.316625, 3.316625] +
                    [2.828427, 3.605551, 3.605551, 3.741657, 3.741657] +
                    [1., 2., 2.236068, 2.236068, 1.414214],
                "expected_with_subtrees":
                    [1.      , 2.      , 2.236068, 2.236068, 1.414214] +
                    [1.414214, 3.162278, 3.316625, 3.316625] +
                    [1.414214, 2.236068, 2.236068, 2.44949 , 2.44949] +
                    [1., 2., 2.236068, 2.236068, 1.414214],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [1.      , 2.      , 2.236068, 2.236068, 1.414214] +
                    [1.414214, 3.162278, 3.316625, 3.316625] +
                    [2.828427, 3.605551, 3.605551, 3.741657, 3.741657],
                "expected_with_subtrees":
                    [1.      , 2.      , 2.236068, 2.236068, 1.414214] +
                    [1.414214, 3.162278, 3.316625, 3.316625],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414214, 2.236068, 2.236068, 2.44949 , 2.44949],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1., 2., 2.236068, 2.236068, 1.414214],
                "expected_with_subtrees": [1., 2., 2.236068, 2.236068, 1.414214],
            }

        ],
        "section_term_radial_distances": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [2.236068, 2.236068, 1.414214] +
                    [3.316625, 3.316625] +
                    [3.605551, 3.741657, 3.741657] +
                    [2.236068, 2.236068, 1.414214],
                "expected_with_subtrees":
                    [2.236068, 2.236068, 1.414214] +
                    [3.316625, 3.316625] +
                    [2.236068, 2.44949 , 2.44949] +
                    [2.236068, 2.236068, 1.414214],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [2.236068, 2.236068, 1.414214] +
                    [3.316625, 3.316625] +
                    [3.605551, 3.741657, 3.741657],
                "expected_with_subtrees":
                    [2.236068, 2.236068, 1.414214] +
                    [3.316625, 3.316625],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2.236068, 2.44949 , 2.44949],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [2.236068, 2.236068, 1.414214],
                "expected_with_subtrees": [2.236068, 2.236068, 1.414214],
            }

        ],
        "section_bif_radial_distances": [
            {
                # radial distances change when the mixed subtrees are processed because
                # the root of the subtree is considered instead of the tree root
                # heterogeneous forks are not valid forking points
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1., 2., 1.414214, 3.162278, 2.828427, 3.605551, 1., 2.],
                "expected_with_subtrees":
                    [1., 2., 3.162278, 1.414214, 2.236068, 1., 2.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [1., 2., 1.414214, 3.162278, 2.828427, 3.605551],
                "expected_with_subtrees": [1., 2., 3.162278],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414214, 2.236068],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1., 2.],
                "expected_with_subtrees": [1., 2.],
            }
        ],
        "section_end_distances": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1.] * 5 +
                    [1.414214, 2., 1., 1.] +
                    [1.414214, 1., 1., 1., 1.] +
                    [1.] * 5,
                "expected_with_subtrees":
                    [1.] * 5 +
                    [1.414214, 2., 1., 1.] +
                    [1.414214, 1., 1., 1., 1.] +
                    [1.] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [1.] * 5 +
                    [1.414214, 2., 1., 1.] +
                    [1.414214, 1., 1., 1., 1.],
                "expected_with_subtrees":
                    [1.] * 5 +
                    [1.414214, 2., 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414214, 1., 1., 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.] * 5,
                "expected_with_subtrees": [1.] * 5,
            }
        ],
        "section_term_lengths": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1.] * 11,
                "expected_with_subtrees": [1.] * 11,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1.] * 8,
                "expected_with_subtrees": [1.] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.] * 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.] * 3,
                "expected_with_subtrees": [1.] * 3,
            }
        ],
        "section_taper_rates": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.0] * 19,
                "expected_with_subtrees": [0.0] * 19,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.0] * 14,
                "expected_with_subtrees": [0.0] * 9,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.0] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.0] * 5,
                "expected_with_subtrees": [0.0] * 5,
            }
        ],
        "section_bif_lengths": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1., 1., 1.414214, 2., 1.414214, 1., 1., 1.],
                "expected_with_subtrees":
                    [1., 1., 2., 1.414214, 1., 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1., 1., 1.414214, 2., 1.414214, 1.],
                "expected_with_subtrees": [1., 1., 2.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414214, 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1., 1.],
                "expected_with_subtrees": [1., 1.],
            },
        ],
        "section_branch_orders": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [0, 1, 2, 2, 1, 0, 1, 2, 2, 1, 2, 2, 3, 3, 0, 1, 2, 2, 1],
                "expected_with_subtrees":
                    [0, 1, 2, 2, 1, 0, 1, 2, 2, 1, 2, 2, 3, 3, 0, 1, 2, 2, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0, 1, 2, 2, 1, 0, 1, 2, 2, 1, 2, 2, 3, 3],
                "expected_with_subtrees": [0, 1, 2, 2, 1, 0, 1, 2, 2],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1, 2, 2, 3, 3],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0, 1, 2, 2, 1],
                "expected_with_subtrees": [0, 1, 2, 2, 1],
            },
        ],
        "section_bif_branch_orders": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0, 1, 0, 1, 1, 2, 0, 1],
                "expected_with_subtrees": [0, 1, 1, 1, 2, 0, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0, 1, 0, 1, 1, 2],
                "expected_with_subtrees": [0, 1, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1, 2],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0, 1],
                "expected_with_subtrees": [0, 1],
            },
        ],
        "section_term_branch_orders": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [2, 2, 1, 2, 2, 2, 3, 3, 2, 2, 1],
                "expected_with_subtrees": [2, 2, 1, 2, 2, 2, 3, 3, 2, 2, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [2, 2, 1, 2, 2, 2, 3, 3],
                "expected_with_subtrees": [2, 2, 1, 2, 2],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2, 3, 3],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [2, 2, 1],
                "expected_with_subtrees": [2, 2, 1],
            },
        ],
        "section_strahler_orders": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [2, 2, 1, 1, 1, 3, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1],
                "expected_with_subtrees":
                    [2, 2, 1, 1, 1, 3, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [2, 2, 1, 1, 1, 3, 2, 1, 1, 2, 1, 2, 1, 1],
                "expected_with_subtrees": [2, 2, 1, 1, 1, 3, 2, 1, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2, 1, 2, 1, 1],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [2, 2, 1, 1, 1],
                "expected_with_subtrees": [2, 2, 1, 1, 1],
            },
        ],
        "segment_lengths": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1.] * 5 +
                    [1.414214, 2., 1., 1.] +
                    [1.414214, 1., 1., 1., 1.] +
                    [1.] * 5,
                "expected_with_subtrees":
                    [1.] * 5 +
                    [1.414214, 2., 1., 1.] +
                    [1.414214, 1., 1., 1., 1.] +
                    [1.] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [1.] * 5 +
                    [1.414214, 2., 1., 1.] +
                    [1.414214, 1., 1., 1., 1.],
                "expected_with_subtrees":
                    [1.] * 5 +
                    [1.414214, 2., 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414214, 1., 1., 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.] * 5,
                "expected_with_subtrees": [1.] * 5,
            }
        ],
        "segment_areas": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [0.628319] * 5 +
                    [0.888577, 1.256637, 0.628319, 0.628319] +
                    [0.888577, 0.628319, 0.628319, 0.628319, 0.628319] +
                    [0.628319] * 5,
                "expected_with_subtrees":
                    [0.628319] * 5 +
                    [0.888577, 1.256637, 0.628319, 0.628319] +
                    [0.888577, 0.628319, 0.628319, 0.628319, 0.628319] +
                    [0.628319] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [0.628319] * 5 +
                    [0.888577, 1.256637, 0.628319, 0.628319] +
                    [0.888577, 0.628319, 0.628319, 0.628319, 0.628319],
                "expected_with_subtrees":
                    [0.628319] * 5 +
                    [0.888577, 1.256637, 0.628319, 0.628319],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees":
                    [0.888577, 0.628319, 0.628319, 0.628319, 0.628319],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.628318] * 5,
                "expected_with_subtrees": [0.628318] * 5,
            }
        ],
        "segment_volumes": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [0.031415] * 5 +
                    [0.044429, 0.062832, 0.031416, 0.031416] +
                    [0.044429, 0.031416, 0.031416, 0.031416, 0.031416] +
                    [0.031416] * 5,
                "expected_with_subtrees":
                    [0.031415] * 5 +
                    [0.044429, 0.062832, 0.031416, 0.031416] +
                    [0.044429, 0.031416, 0.031416, 0.031416, 0.031416] +
                    [0.031416] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [0.031415] * 5 +
                    [0.044429, 0.062832, 0.031416, 0.031416] +
                    [0.044429, 0.031416, 0.031416, 0.031416, 0.031416],
                "expected_with_subtrees":
                    [0.031415] * 5 +
                    [0.044429, 0.062832, 0.031416, 0.031416],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees":
                    [0.044429, 0.031416, 0.031416, 0.031416, 0.031416],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.031415] * 5,
                "expected_with_subtrees": [0.031415] * 5,
            }
        ],
        "segment_radii": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.1] * 19,
                "expected_with_subtrees": [0.1] * 19,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.1] * 14,
                "expected_with_subtrees": [0.1] * 9,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.1] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.1] * 5,
                "expected_with_subtrees": [0.1] * 5,
            }
        ],
        "segment_taper_rates": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [0.0] * 19,
                "expected_with_subtrees": [0.0] * 19,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [0.0] * 14,
                "expected_with_subtrees": [0.0] * 9,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.0] * 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.0] * 5,
                "expected_with_subtrees": [0.0] * 5,
            },
        ],
        "segment_radial_distances": [
            {
                # radial distances change when the mixed subtrees are processed because
                # the root of the subtree is considered
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [0.5, 1.5, 2.061553, 2.061553, 1.118034] +
                    [0.707107, 2.236068, 3.201562, 3.201562] +
                    [2.12132 , 3.201562, 3.201562, 3.640055, 3.640055] +
                    [0.5, 1.5, 2.061553, 2.061553, 1.118034],
                "expected_with_subtrees":
                    [0.5, 1.5, 2.061553, 2.061553, 1.118034] +
                    [0.707107, 2.236068, 3.201562, 3.201562] +
                    [0.707107, 1.802776, 1.802776, 2.291288, 2.291288] +
                    [0.5, 1.5, 2.061553, 2.061553, 1.118034],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [0.5, 1.5, 2.061553, 2.061553, 1.118034] +
                    [0.707107, 2.236068, 3.201562, 3.201562] +
                    [2.12132 , 3.201562, 3.201562, 3.640055, 3.640055],
                "expected_with_subtrees":
                    [0.5, 1.5, 2.061553, 2.061553, 1.118034] +
                    [0.707107, 2.236068, 3.201562, 3.201562],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.707107, 1.802776, 1.802776, 2.291288, 2.291288],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [0.5, 1.5, 2.061553, 2.061553, 1.118034],
                "expected_with_subtrees": [0.5, 1.5, 2.061553, 2.061553, 1.118034],
            },
        ],
        "segment_midpoints": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [
                    [-1.5, 0.0, 0.0], [-2.5, 0.0, 0.0], [-3.0, 0.0, 0.5], [-3.0, 0.0, -0.5],
                    [-2.0, 0.5, 0.0], [0.5, 1.5, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.5],
                    [1.0, 4.0, -0.5], [1.5, 2.5, 0.0], [2.0, 3.5, 0.0], [2.5, 3.0, 0.0],
                    [3.0, 3.0, 0.5], [3.0, 3.0, -0.5], [0.0, -1.5, 0.0], [0.0, -2.5, 0.0],
                    [0.0, -3.0, 0.5], [0.0, -3.0, -0.5], [0.5, -2.0, 0.0]],
                "expected_with_subtrees": [
                    [-1.5, 0.0, 0.0], [-2.5, 0.0, 0.0], [-3.0, 0.0, 0.5], [-3.0, 0.0, -0.5],
                    [-2.0, 0.5, 0.0], [0.5, 1.5, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.5],
                    [1.0, 4.0, -0.5], [1.5, 2.5, 0.0], [2.0, 3.5, 0.0], [2.5, 3.0, 0.0],
                    [3.0, 3.0, 0.5], [3.0, 3.0, -0.5], [0.0, -1.5, 0.0], [0.0, -2.5, 0.0],
                    [0.0, -3.0, 0.5], [0.0, -3.0, -0.5], [0.5, -2.0, 0.0]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [
                    [-1.5, 0.0, 0.0], [-2.5, 0.0, 0.0], [-3.0, 0.0, 0.5], [-3.0, 0.0, -0.5],
                    [-2.0, 0.5, 0.0], [0.5, 1.5, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.5],
                    [1.0, 4.0, -0.5], [1.5, 2.5, 0.0], [2.0, 3.5, 0.0], [2.5, 3.0, 0.0],
                    [3.0, 3.0, 0.5], [3.0, 3.0, -0.5]],
                "expected_with_subtrees": [
                    [-1.5, 0.0, 0.0], [-2.5, 0.0, 0.0], [-3.0, 0.0, 0.5], [-3.0, 0.0, -0.5],
                    [-2.0, 0.5, 0.0], [0.5, 1.5, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.5],
                    [1.0, 4.0, -0.5]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [
                    [1.5, 2.5, 0.0], [2.0, 3.5, 0.0], [2.5, 3.0, 0.0],
                    [3.0, 3.0, 0.5], [3.0, 3.0, -0.5]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [
                    [0.0, -1.5, 0.0], [0.0, -2.5, 0.0], [0.0, -3.0, 0.5],
                    [0.0, -3.0, -0.5], [0.5, -2.0, 0.0]],
                "expected_with_subtrees": [
                    [0.0, -1.5, 0.0], [0.0, -2.5, 0.0], [0.0, -3.0, 0.5],
                    [0.0, -3.0, -0.5], [0.5, -2.0, 0.0]],
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
                "expected_wout_subtrees": 19,
                "expected_with_subtrees": 19,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 14,
                "expected_with_subtrees": 9,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0,
                "expected_with_subtrees": 5,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 5,
                "expected_with_subtrees": 5,
            },
        ],
        "number_of_bifurcations": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 8,
                "expected_with_subtrees": 7,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 6,
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
        "number_of_forking_points": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 8,
                "expected_with_subtrees": 7,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 6,
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
        "local_bifurcation_angles": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1.570796, 3.141593, 0.785398, 3.141593,
                     1.570796, 3.141593, 1.570796, 3.141593],
                "expected_with_subtrees":
                    [1.570796, 3.141593, 3.141593, 1.570796, 3.141593, 1.570796, 3.141593],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [1.570796, 3.141593, 0.785398, 3.141593, 1.570796, 3.141593],
                "expected_with_subtrees": [1.570796, 3.141593, 3.141593],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.570796, 3.141593],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.570796, 3.141593],
                "expected_with_subtrees": [1.570796, 3.141593],
            },
        ],
        "remote_bifurcation_angles": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1.570796, 3.141593, 0.785398, 3.141593,
                     1.570796, 3.141593, 1.570796, 3.141593],
                "expected_with_subtrees":
                    [1.570796, 3.141593, 3.141593, 1.570796, 3.141593, 1.570796, 3.141593],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [1.570796, 3.141593, 0.785398, 3.141593, 1.570796, 3.141593],
                "expected_with_subtrees": [1.570796, 3.141593, 3.141593],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.570796, 3.141593],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.570796, 3.141593],
                "expected_with_subtrees": [1.570796, 3.141593],
            },
        ],
        "sibling_ratios": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [1.0] * 8,
                "expected_with_subtrees": [1.0] * 7,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [1.0] * 6,
                "expected_with_subtrees": [1.0] * 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.0] * 2,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.0] * 2,
                "expected_with_subtrees": [1.0] * 2,
            },
        ],
        "partition_pairs": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [[3.0, 1.0], [1.0, 1.0], [3.0, 5.0],
                     [1.0, 1.0], [1.0, 3.0], [1.0, 1.0], [3.0, 1.0], [1.0, 1.0]],
                "expected_with_subtrees":
                    [[3.0, 1.0], [1.0, 1.0], [1.0, 1.0],
                     [1.0, 3.0], [1.0, 1.0], [3.0, 1.0], [1.0, 1.0]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [[3.0, 1.0], [1.0, 1.0], [3.0, 5.0], [1.0, 1.0], [1.0, 3.0], [1.0, 1.0]],
                "expected_with_subtrees": [[3.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [[1.0, 3.0], [1.0, 1.0]],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [[3.0, 1.0], [1.0, 1.0]],
                "expected_with_subtrees": [[3.0, 1.0], [1.0, 1.0]],
            },
        ],
        "diameter_power_relations": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [2.0] * 8,
                "expected_with_subtrees": [2.0] * 7,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [2.0] * 6,
                "expected_with_subtrees": [2.0] * 3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2.0] * 2,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [2.0] * 2,
                "expected_with_subtrees": [2.0] * 2,
            },
        ],
        "bifurcation_partitions": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [3., 1., 1.666667, 1., 3., 1., 3., 1.],
                "expected_with_subtrees": [3., 1., 1., 3., 1., 3., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [3., 1., 1.666667, 1., 3., 1. ],
                "expected_with_subtrees": [3., 1., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [3., 1.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [3., 1.],
                "expected_with_subtrees": [3., 1.],
            },
        ],
        "section_path_distances": [
            {
                # subtree path distances are calculated to the root of the subtree
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [
                    1.0, 2.0, 3.0, 3.0, 2.0, 1.414213, 3.414213, 4.414213,
                    4.414213, 2.828427, 3.828427, 3.828427, 4.828427, 4.828427,
                    1.0, 2.0, 3.0, 3.0, 2.0
                ],
                "expected_with_subtrees": [
                    1., 2., 3., 3., 2., 1.414214, 3.414214, 4.414214, 4.414214, 1.414214,
                    2.414214, 2.414214, 3.414214, 3.414214, 1., 2., 3., 3., 2.
                ],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [
                    1., 2., 3., 3., 2., 1.414214, 3.414214, 4.414214, 4.414214,
                    2.828427, 3.828427, 3.828427, 4.828427, 4.828427
                ],
                "expected_with_subtrees":
                    [1., 2., 3., 3., 2., 1.414214, 3.414214, 4.414214, 4.414214],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [1.414214, 2.414214, 2.414214, 3.414214, 3.414214],
            },
        ],
        "terminal_path_lengths": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [3., 3., 2., 4.414214, 4.414214, 3.828427, 4.828427, 4.828427, 3., 3., 2.],
                "expected_with_subtrees":
                    [3., 3., 2., 4.414214, 4.414214, 2.414214, 3.414214, 3.414214, 3., 3., 2.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [3., 3., 2., 4.414214, 4.414214, 3.828427, 4.828427, 4.828427],
                "expected_with_subtrees": [3., 3., 2., 4.414214, 4.414214],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2.414214, 3.414214, 3.414214],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [3., 3., 2.],
                "expected_with_subtrees": [3., 3., 2.],
            },
        ],
        "principal_direction_extents": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": [3.321543, 5.470702, 3.421831],
                "expected_with_subtrees": [3.321543, 2.735383, 3.549779, 3.421831],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [3.321543, 5.470702],
                "expected_with_subtrees": [3.321543, 2.735383],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [3.549779],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [3.421831],
                "expected_with_subtrees": [3.421831],
            },
        ],
        "partition_asymmetry": [
            {
                "kwargs": {
                    "neurite_type": NeuriteType.all,
                    "variant": "branch-order",
                    "method": "petilla",
                },
                "expected_wout_subtrees": [0.5, 0.0, 0.25, 0.0, 0.5, 0.0, 0.5, 0.0],
                "expected_with_subtrees": [0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
            },
            {
                "kwargs": {
                    "neurite_type": NeuriteType.basal_dendrite,
                    "variant": "branch-order",
                    "method": "petilla",
                },
                "expected_wout_subtrees": [0.5, 0.0, 0.25, 0.0, 0.5, 0.0],
                "expected_with_subtrees": [0.5, 0.0, 0.0],
            },
            {
                "kwargs": {
                    "neurite_type": NeuriteType.axon,
                    "variant": "branch-order",
                    "method": "petilla",
                },
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.5, 0.0],
            },
            {
                "kwargs": {
                    "neurite_type": NeuriteType.apical_dendrite,
                    "variant": "branch-order",
                    "method": "petilla",
                },
                "expected_wout_subtrees": [0.5, 0.0],
                "expected_with_subtrees": [0.5, 0.0],
            },
            {
                "kwargs": {
                    "neurite_type": NeuriteType.all,
                    "variant": "length",
                },
                "expected_wout_subtrees": [0.4, 0.0, 0.130601, 0.0, 0.184699, 0.0, 0.4, 0.0],
                "expected_with_subtrees": [0.4, 0.0, 0.0, 0.369398, 0.0, 0.4, 0.0],
            },
            {
                "kwargs": {
                    "neurite_type": NeuriteType.basal_dendrite,
                    "variant": "length",
                },
                "expected_wout_subtrees": [0.4, 0.0, 0.130601, 0.0, 0.184699, 0.0],
                "expected_with_subtrees": [0.4, 0.0, 0.0],
            },
            {
                "kwargs": {
                    "neurite_type": NeuriteType.axon,
                    "variant": "length",
                },
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.369398, 0.0],
            },
            {
                "kwargs": {
                    "neurite_type": NeuriteType.apical_dendrite,
                    "variant": "length",
                },
                "expected_wout_subtrees": [0.4, 0.0],
                "expected_with_subtrees": [0.4, 0.0],
            },
        ],
        "partition_asymmetry_length": [
            {
                "kwargs": {
                    "neurite_type": NeuriteType.all,
                },
                "expected_wout_subtrees": [0.4, 0.0, 0.130601, 0.0, 0.184699, 0.0, 0.4, 0.0],
                "expected_with_subtrees": [0.4, 0.0, 0.0, 0.369398, 0.0, 0.4, 0.0],
            },
            {
                "kwargs": {
                    "neurite_type": NeuriteType.basal_dendrite,
                },
                "expected_wout_subtrees": [0.4, 0.0, 0.130601, 0.0, 0.184699, 0.0],
                "expected_with_subtrees": [0.4, 0.0, 0.0],
            },
            {
                "kwargs": {
                    "neurite_type": NeuriteType.axon,
                },
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0.369398, 0.0],
            },
            {
                "kwargs": {
                    "neurite_type": NeuriteType.apical_dendrite,
                },
                "expected_wout_subtrees": [0.4, 0.0],
                "expected_with_subtrees": [0.4, 0.0],
            },
        ]
    }

    features_not_tested = list(
        set(_NEURITE_FEATURES) - set(features.keys()) - set(_MORPHOLOGY_FEATURES)
    )

    #assert not features_not_tested, (
    #    "The following morphology tests need to be included in the tests:\n\n" +
    #    "\n".join(sorted(features_not_tested)) + "\n"
    #)

    return _dispatch_features(features, mode)

'''
@pytest.mark.parametrize(
    "feature_name, kwargs, expected", _neurite_features(mode="wout-subtrees")
)
def test_morphology__neurite_features_wout_subtrees(feature_name, kwargs, expected, mixed_morph):
    _assert_feature_equal(mixed_morph, feature_name, expected, kwargs, use_subtrees=False)
'''

@pytest.mark.parametrize(
    "feature_name, kwargs, expected", _neurite_features(mode="with-subtrees")
)
def test_morphology__neurite_features_with_subtrees(feature_name, kwargs, expected, mixed_morph):
    _assert_feature_equal(mixed_morph, feature_name, expected, kwargs, use_subtrees=True)

