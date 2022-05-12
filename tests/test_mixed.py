import sys
import warnings
import pytest
import neurom
import numpy as np
import pandas as pd
import numpy.testing as npt
from neurom import NeuriteType
from neurom.features import get
from neurom.core import Population
from neurom.features import _POPULATION_FEATURES, _MORPHOLOGY_FEATURES, _NEURITE_FEATURES
import collections.abc

from neurom.core.types import tree_type_checker as is_type

import neurom.core.morphology
import neurom.features.neurite
import neurom.apps.morph_stats


@pytest.fixture
def mixed_morph():
    """
                                                               (1, 4, 1)
                                                                   |
                                                             S7:B  |
                                                                   |
                                                (1, 4, -1)-----(1, 4, 0)    (2, 4, 0)     (3, 3, 1)
                                                           S8:B    |            |             |
                                                                   |     S10:A  |      S12:A  |
                                                                   |            |   S11:A     |
                                                             S6:B  |        (2, 3, 0)-----(3, 3, 0)
                                                                   |            /             |
                                                                   |   S9:A  /         S13:A  |
                                                                   |      /                   |
                                                               (1, 2, 0)                  (3, 3, -1)
                                                                  /
                                                        S5:B   /
                                                            /       Axon on basal dendrite
    (-3, 0, 1)     (-2, 1, 0)                    (0, 1, 0)
         |              |
      S2 |           S4 |
         |     S1       |     S0
    (-3, 0, 0)-----(-2, 0, 0)-----(-1, 0, 0)     (0, 0, 0) Soma
         |
      S3 |           Basal Dendrite
         |
    (-3, 0, -1)                                 (0, -1, 0)
                                                     |
                                                S14  |
                                                     |     S17
                            Apical Dendrite     (0, -2, 0)-----(1, -2, 0)
                                                     |
                                                S15  |
                                            S17      |     S16
                                (0, -3, -1)-----(0, -3, 0)-----(0, -3, 1)

    basal_dendrite: homogeneous
        section ids: [0, 1, 2, 3, 4]

    axon_on_basal_dendrite: heterogeneous
        section_ids:
            - basal: [5, 6, 7, 8]
            - axon : [9, 10, 11, 12, 13]

    apical_dendrite: homogeneous:
        section_ids: [14, 15, 16, 17, 18]
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

@pytest.fixture
def three_types_neurite_morph():
    return neurom.load_morphology(
    """
    1    1   0  0  0   0.5 -1
    2    3   0  1  0   0.1  1
    3    3   1  2  0   0.1  2
    4   3   1  4  0   0.1  3
    5   3   1  4  1   0.1  4
    6   3   1  4 -1   0.1  4
    7   2   2  3  0   0.1  3
    8   2   2  4  0   0.1  7
    9   2   3  3  0   0.1  7
    10   2   3  3  1   0.1  9
    11   4   3  3 -1   0.1  9
    """,
    reader="swc")

def test_heterogeneous_neurites(mixed_morph):

    assert not mixed_morph.neurites[0].is_heterogeneous()
    assert mixed_morph.neurites[1].is_heterogeneous()
    assert not mixed_morph.neurites[2].is_heterogeneous()


def test_is_homogeneous_point(mixed_morph):

    heterogeneous_neurite = mixed_morph.neurites[1]

    sections = list(heterogeneous_neurite.iter_sections())

    # first section has one axon and one basal children
    assert not sections[0].is_homogeneous_point()

    # second section is pure basal
    assert sections[1].is_homogeneous_point()


def test_homogeneous_subtrees(mixed_morph, three_types_neurite_morph):

    basal, axon_on_basal, apical = mixed_morph.neurites

    assert neurom.core.morphology._homogeneous_subtrees(basal) == [basal]

    sections = list(axon_on_basal.iter_sections())

    subtrees = neurom.core.morphology._homogeneous_subtrees(axon_on_basal)

    assert subtrees[0].root_node.id == axon_on_basal.root_node.id
    assert subtrees[0].root_node.type == NeuriteType.basal_dendrite

    assert subtrees[1].root_node.id == sections[4].id
    assert subtrees[1].root_node.type == NeuriteType.axon

    with pytest.warns(
        UserWarning,
        match="Neurite <type: NeuriteType.basal_dendrite> is not an axon-carrying dendrite."
    ):
        three_types_neurite, = three_types_neurite_morph.neurites
        neurom.core.morphology._homogeneous_subtrees(three_types_neurite)


def test_iter_neurites__heterogeneous(mixed_morph):

    subtrees = list(neurom.core.morphology.iter_neurites(mixed_morph, use_subtrees=False))

    assert len(subtrees) == 3
    assert subtrees[0].type == NeuriteType.basal_dendrite
    assert subtrees[1].type == NeuriteType.basal_dendrite
    assert subtrees[2].type == NeuriteType.apical_dendrite

    subtrees =  list(neurom.core.morphology.iter_neurites(mixed_morph, use_subtrees=True))

    assert len(subtrees) == 4
    assert subtrees[0].type == NeuriteType.basal_dendrite
    assert subtrees[1].type == NeuriteType.basal_dendrite
    assert subtrees[2].type == NeuriteType.axon
    assert subtrees[3].type == NeuriteType.apical_dendrite


def test_core_iter_sections__heterogeneous(mixed_morph):

    def assert_sections(neurite, section_type, expected_section_ids):

        it = neurom.core.morphology.iter_sections(neurite, section_filter=is_type(section_type))
        assert [s.id for s in it] == expected_section_ids

    basal, axon_on_basal, apical = mixed_morph.neurites

    assert_sections(basal, NeuriteType.all, [0, 1, 2, 3, 4])
    assert_sections(basal, NeuriteType.basal_dendrite, [0, 1, 2, 3, 4])
    assert_sections(basal, NeuriteType.axon, [])

    assert_sections(axon_on_basal, NeuriteType.all, [5, 6, 7, 8, 9, 10, 11, 12, 13])
    assert_sections(axon_on_basal, NeuriteType.basal_dendrite, [5, 6, 7, 8])
    assert_sections(axon_on_basal, NeuriteType.axon, [9, 10, 11, 12, 13])

    assert_sections(apical, NeuriteType.all, [14, 15, 16, 17, 18])
    assert_sections(apical, NeuriteType.apical_dendrite, [14, 15, 16, 17, 18])


def test_features_neurite_map_sections__heterogeneous(mixed_morph):

    def assert_sections(neurite, section_type, iterator_type, expected_section_ids):
        function = lambda section: section.id
        section_ids = neurom.features.neurite._map_sections(
            function, neurite, iterator_type=iterator_type, section_type=section_type
        )
        assert section_ids == expected_section_ids

    basal, axon_on_basal, apical = mixed_morph.neurites

    # homogeneous tree, no difference between all and basal_dendrite types.
    assert_sections(
        basal, NeuriteType.all, neurom.core.morphology.Section.ibifurcation_point,
        [0, 1],
    )
    assert_sections(
        basal, NeuriteType.basal_dendrite, neurom.core.morphology.Section.ibifurcation_point,
        [0, 1],
    )
    # heterogeneous tree, forks cannot be heterogeneous if a type other than all is specified
    # Section with id 5 is the transition section, which has a basal and axon children sections
    assert_sections(
        axon_on_basal, NeuriteType.all, neurom.core.morphology.Section.ibifurcation_point,
        [5, 6, 9, 11],
    )
    assert_sections(
        axon_on_basal, NeuriteType.basal_dendrite,
        neurom.core.morphology.Section.ibifurcation_point,
        [6],
    )
    assert_sections(
        axon_on_basal, NeuriteType.axon,
        neurom.core.morphology.Section.ibifurcation_point,
        [9, 11],
    )
    # homogeneous tree, no difference between all and basal_dendrite types.
    assert_sections(
        apical, NeuriteType.all, neurom.core.morphology.Section.ibifurcation_point,
        [14, 15],
    )
    assert_sections(
        apical, NeuriteType.apical_dendrite, neurom.core.morphology.Section.ibifurcation_point,
        [14, 15],
    )


def test_mixed_morph_stats(mixed_morph):

    def assert_stats_equal(actual_dict, expected_dict):
        assert actual_dict.keys() == expected_dict.keys()
        for (key, value) in actual_dict.items():
            expected_value = expected_dict[key]
            if value is None or expected_value is None:
                assert expected_value is value
            else:
                npt.assert_almost_equal(value, expected_value, decimal=3, err_msg=f"\nKey: {key}")

    cfg = {
        'neurite': {
            'max_radial_distance': ['mean'],
            'number_of_sections': ['min'],
            'number_of_bifurcations': ['max'],
            'number_of_leaves': ['median'],
            'total_length': ['min'],
            'total_area': ['max'],
            'total_volume': ['median'],
            'section_lengths': ['mean'],
            'section_term_lengths': ['mean'],
            'section_bif_lengths': ['mean'],
            'section_branch_orders': ['mean'],
            'section_bif_branch_orders': ['mean'],
            'section_term_branch_orders': ['mean'],
            'section_path_distances': ['mean'],
            'section_taper_rates': ['median'],
            'local_bifurcation_angles': ['mean'],
            'remote_bifurcation_angles': ['mean'],
            'partition_asymmetry': ['mean'],
            'partition_asymmetry_length': ['mean'],
            'sibling_ratios': ['mean'],
            'diameter_power_relations': ['median'],
            'section_radial_distances': ['mean'],
            'section_term_radial_distances': ['mean'],
            'section_bif_radial_distances': ['mean'],
            'terminal_path_lengths': ['mean'],
            'section_volumes': ['min'],
            'section_areas': ['mean'],
            'section_tortuosity': ['mean'],
            'section_strahler_orders': ['min']
        },
        'morphology': {
            'soma_surface_area': ['mean'],
            'soma_radius': ['max'],
            'max_radial_distance': ['mean'],
            'number_of_sections_per_neurite': ['median'],
            'total_length_per_neurite': ['mean'],
            'total_area_per_neurite': ['mean'],
            'total_volume_per_neurite': ['mean'],
            'number_of_neurites': ['median']
        },
        'neurite_type': ['AXON', 'BASAL_DENDRITE', 'APICAL_DENDRITE']
    }

    res = neurom.apps.morph_stats.extract_stats(mixed_morph, cfg, use_subtrees=False)

    expected_axon_wout_subtrees = {
        'max_number_of_bifurcations': 0,
        'max_total_area': 0,
        'mean_local_bifurcation_angles': None,
        'mean_max_radial_distance': 0.0,
        'mean_partition_asymmetry': None,
        'mean_partition_asymmetry_length': None,
        'mean_remote_bifurcation_angles': None,
        'mean_section_areas': None,
        'mean_section_bif_branch_orders': None,
        'mean_section_bif_lengths': None,
        'mean_section_bif_radial_distances': None,
        'mean_section_branch_orders': None,
        'mean_section_lengths': None,
        'mean_section_path_distances': None,
        'mean_section_radial_distances': None,
        'mean_section_term_branch_orders': None,
        'mean_section_term_lengths': None,
        'mean_section_term_radial_distances': None,
        'mean_section_tortuosity': None,
        'mean_sibling_ratios': None,
        'mean_terminal_path_lengths': None,
        'median_diameter_power_relations': None,
        'median_number_of_leaves': 0,
        'median_section_taper_rates': None,
        'median_total_volume': 0,
        'min_number_of_sections': 0,
        'min_section_strahler_orders': None,
        'min_section_volumes': None,
        'min_total_length': 0
    }

    assert_stats_equal(res["axon"], expected_axon_wout_subtrees)

    res_df = neurom.apps.morph_stats.extract_dataframe(mixed_morph, cfg, use_subtrees=False)

    # get axon column and tranform it to look like the expected values above
    values = res_df.loc[pd.IndexSlice[:, "axon"]].iloc[0, :].to_dict()
    assert_stats_equal(values, expected_axon_wout_subtrees)


    res = neurom.apps.morph_stats.extract_stats(mixed_morph, cfg, use_subtrees=True)

    expected_axon_with_subtrees = {
        'max_number_of_bifurcations': 2,
        'max_total_area': 3.4018507611950346,
        'mean_local_bifurcation_angles': 2.356194490192345,
        'mean_max_radial_distance': 4.472136,
        'mean_partition_asymmetry': 0.25,
        'mean_partition_asymmetry_length': 0.1846990320847273,
        'mean_remote_bifurcation_angles': 2.356194490192345,
        'mean_section_areas': 0.6803701522390069,
        'mean_section_bif_branch_orders': 1.5,
        'mean_section_bif_lengths': 1.2071068,
        'mean_section_bif_radial_distances': 3.9240959,
        'mean_section_branch_orders': 2.2,
        'mean_section_lengths': 1.0828427,
        'mean_section_path_distances': 2.614213538169861,
        'mean_section_radial_distances': 4.207625,
        'mean_section_term_branch_orders': 2.6666666666666665,
        'mean_section_term_lengths': 1.0,
        'mean_section_term_radial_distances': 4.396645,
        'mean_section_tortuosity': 1.0,
        'mean_sibling_ratios': 1.0,
        'mean_terminal_path_lengths': 3.0808802048365274,
        'median_diameter_power_relations': 2.0,
        'median_number_of_leaves': 3,
        'median_section_taper_rates': 8.6268466e-17,
        'median_total_volume': 0.17009254152367845,
        'min_number_of_sections': 5,
        'min_section_strahler_orders': 1,
        'min_section_volumes': 0.03141592778425469,
        'min_total_length': 5.414213538169861
    }

    assert_stats_equal(res["axon"], expected_axon_with_subtrees)

    res_df = neurom.apps.morph_stats.extract_dataframe(mixed_morph, cfg, use_subtrees=True)

    # get axon column and tranform it to look like the expected values above
    values = res_df.loc[pd.IndexSlice[:, "axon"]].iloc[0, :].to_dict()
    assert_stats_equal(values, expected_axon_with_subtrees)


@pytest.fixture
def population(mixed_morph):
    return Population([mixed_morph, mixed_morph])


def _assert_feature_equal(values, expected_values, per_neurite=False):

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

    def check(values, expected_values):
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

    if per_neurite:

        assert len(values) == len(expected_values)
        for neurite_values, expected_neurite_values in zip(values, expected_values):
            check(neurite_values, expected_neurite_values)
    else:
        check(values, expected_values)



def _dispatch_features(features, mode=None):
    for feature_name, configurations in features.items():

        for cfg in configurations:
            kwargs = cfg["kwargs"] if "kwargs" in cfg else {}

            if mode == "with-subtrees":
                expected = cfg["expected_with_subtrees"]
            elif mode == "wout-subtrees":
                expected = cfg["expected_wout_subtrees"]
            else:
                expected = cfg["expected"]

            yield feature_name, kwargs, expected


def _population_features(mode):

    features = {
        "sholl_frequency": [
            {
                "kwargs": {"neurite_type": NeuriteType.all, "step_size": 3},
                "expected_wout_subtrees": [0, 4],
                "expected_with_subtrees": [0, 4],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite, "step_size": 3},
                "expected_wout_subtrees": [0, 4],
                "expected_with_subtrees": [0, 2],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon, "step_size": 3},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [0, 2],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite, "step_size": 2},
                "expected_wout_subtrees": [0, 2],
                "expected_with_subtrees": [0, 2],
            },

        ],
    }

    features_not_tested = list(
        set(_POPULATION_FEATURES) - set(features.keys())
    )

    assert not features_not_tested, (
        "The following morphology tests need to be included in the tests:\n\n" +
        "\n".join(sorted(features_not_tested)) + "\n"
    )

    return _dispatch_features(features, mode)


@pytest.mark.parametrize("feature_name, kwargs, expected", _population_features(mode="wout-subtrees"))
def test_population__population_features_wout_subtrees(feature_name, kwargs, expected, population):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = get(feature_name, population, use_subtrees=False, **kwargs)
        _assert_feature_equal(values, expected)


@pytest.mark.parametrize("feature_name, kwargs, expected", _population_features(mode="with-subtrees"))
def test_population__population_features_with_subtrees(feature_name, kwargs, expected, population):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = get(feature_name, population, use_subtrees=True, **kwargs)
        _assert_feature_equal(values, expected)


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
                "expected_wout_subtrees": 4.472136,
                "expected_with_subtrees": 4.472136,
            },
            {
                # with a global origin, AoD axon subtree [2, 4] is always furthest from soma
                "kwargs": {"neurite_type": NeuriteType.all, "origin": np.array([0., 0., 0.])},
                "expected_wout_subtrees": 4.472136,
                "expected_with_subtrees": 4.472136,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 4.472136,
                "expected_with_subtrees": 4.24264,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite, "origin": np.array([0., 0., 0.])},
                "expected_wout_subtrees": 4.472136,
                "expected_with_subtrees": 4.242641,

            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": 0.0,
                "expected_with_subtrees": 4.472136,
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
        "aspect_ratio":[
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 0.630311,
                "expected_with_subtrees": 0.630311,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 0.305701,
                "expected_with_subtrees": 0.284467,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": np.nan,
                "expected_with_subtrees": 0.666667,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 0.5,
                "expected_with_subtrees": 0.5,
            },
        ],
        "circularity": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 0.739583,
                "expected_with_subtrees": 0.739583,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 0.525588,
                "expected_with_subtrees": 0.483687,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": np.nan,
                "expected_with_subtrees": 0.544013,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 0.539012,
                "expected_with_subtrees": 0.539012,
            },
        ],
        "shape_factor": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 0.40566,
                "expected_with_subtrees": 0.40566,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 0.21111,
                "expected_with_subtrees": 0.18750,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": np.nan,
                "expected_with_subtrees": 0.3,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": 0.25,
                "expected_with_subtrees": 0.25,
            },
        ],
        "length_fraction_above_soma": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees": 0.567898,
                "expected_with_subtrees": 0.567898,
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": 0.61591,
                "expected_with_subtrees": 0.74729,
            },
        ],
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
                    [2., 3., 3.162278, 3.162278, 2.236068] +
                    [2.236068, 4.123106, 4.24264 , 4.24264] +
                    [3.605551, 4.472136, 4.24264 , 4.358899, 4.358899] +
                    [2., 3., 3.162278, 3.162278, 2.236068],
                "expected_with_subtrees":
                    [2., 3., 3.162278, 3.162278, 2.236068] +
                    [2.236068, 4.123106, 4.24264 , 4.24264] +
                    [3.605551, 4.472136, 4.24264 , 4.358899, 4.358899] +
                    [2., 3., 3.162278, 3.162278, 2.236068],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [2., 3., 3.162278, 3.162278, 2.236068] +
                    [2.236068, 4.123106, 4.24264 , 4.24264] +
                    [3.605551, 4.472136, 4.24264 , 4.358899, 4.358899],
                "expected_with_subtrees":
                    [2., 3., 3.162278, 3.162278, 2.236068] +
                    [2.236068, 4.123106, 4.24264 , 4.24264],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [3.605551, 4.472136, 4.24264 , 4.358899, 4.358899],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [2., 3., 3.162278, 3.162278, 2.236068],
                "expected_with_subtrees": [2., 3., 3.162278, 3.162278, 2.236068],
            }

        ],
        "section_term_radial_distances": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [3.162278, 3.162278, 2.236068] +
                    [4.24264 , 4.24264] +
                    [4.472136, 4.358899, 4.358899] +
                    [3.162278, 3.162278, 2.236068],
                "expected_with_subtrees":
                    [3.162278, 3.162278, 2.236068] +
                    [4.24264 , 4.24264] +
                    [4.472136, 4.358899, 4.358899] +
                    [3.162278, 3.162278, 2.236068],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [3.162278, 3.162278, 2.236068] +
                    [4.24264 , 4.24264] +
                    [4.472136, 4.358899, 4.358899],
                "expected_with_subtrees":
                    [3.162278, 3.162278, 2.236068] +
                    [4.24264 , 4.24264],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [4.472136, 4.358899, 4.358899],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [3.162278, 3.162278, 2.236068],
                "expected_with_subtrees": [3.162278, 3.162278, 2.236068],
            }

        ],
        "section_bif_radial_distances": [
            {
                # radial distances change when the mixed subtrees are processed because
                # the root of the subtree is considered instead of the tree root
                # heterogeneous forks are not valid forking points
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [2., 3., 2.236068, 4.123106, 3.605551, 4.24264 , 2., 3.],
                "expected_with_subtrees":
                    [2., 3., 4.123106, 3.605551, 4.24264 , 2., 3.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [2., 3., 2.236068, 4.123106, 3.605551, 4.24264],
                "expected_with_subtrees": [2., 3., 4.123106],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [3.605551, 4.24264],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [2., 3.],
                "expected_with_subtrees": [2., 3.],
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
                    [1.5, 2.5, 3.041381, 3.041381, 2.061553] +
                    [1.581139, 3.162278, 4.153312, 4.153312] +
                    [2.915476, 4.031129, 3.905125, 4.272002, 4.272002] +
                    [1.5, 2.5, 3.041381, 3.041381, 2.061553],
                "expected_with_subtrees":
                    [1.5, 2.5, 3.041381, 3.041381, 2.061553] +
                    [1.581139, 3.162278, 4.153312, 4.153312] +
                    [2.915476, 4.031129, 3.905125, 4.272002, 4.272002] +
                    [1.5, 2.5, 3.041381, 3.041381, 2.061553],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees":
                    [1.5, 2.5, 3.041381, 3.041381, 2.061553] +
                    [1.581139, 3.162278, 4.153312, 4.153312] +
                    [2.915476, 4.031129, 3.905125, 4.272002, 4.272002],
                "expected_with_subtrees":
                    [1.5, 2.5, 3.041381, 3.041381, 2.061553] +
                    [1.581139, 3.162278, 4.153312, 4.153312],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2.915476, 4.031129, 3.905125, 4.272002, 4.272002],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [1.5, 2.5, 3.041381, 3.041381, 2.061553],
                "expected_with_subtrees": [1.5, 2.5, 3.041381, 3.041381, 2.061553],
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
                "expected_wout_subtrees": [2., 3.596771, 2.],
                "expected_with_subtrees": [2., 3.154926, 2.235207, 2.],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.basal_dendrite},
                "expected_wout_subtrees": [2., 3.596771],
                "expected_with_subtrees": [2., 3.154926],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.axon},
                "expected_wout_subtrees": [],
                "expected_with_subtrees": [2.235207],
            },
            {
                "kwargs": {"neurite_type": NeuriteType.apical_dendrite},
                "expected_wout_subtrees": [2.],
                "expected_with_subtrees": [2.],
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
        ],
        "segment_path_lengths": [
            {
                "kwargs": {"neurite_type": NeuriteType.all},
                "expected_wout_subtrees":
                    [1.0, 2.0, 3.0, 3.0, 2.0] +
                    [1.414213, 3.414213, 4.414213, 4.414213] +
                    [2.828427, 3.828427, 3.828427, 4.828427, 4.828427] +
                    [1.0, 2.0, 3.0, 3.0, 2.0],
                "expected_with_subtrees":
                    [1.0, 2.0, 3.0, 3.0, 2.0] +
                    [1.414213, 3.414213, 4.414213, 4.414213] +
                    [1.414214, 2.414214, 2.414214, 3.414214, 3.414214] +
                    [1.0, 2.0, 3.0, 3.0, 2.0],
            },
        ],
    }

    features_not_tested = (set(_MORPHOLOGY_FEATURES) | set(_NEURITE_FEATURES)) - set(features.keys())

    assert not features_not_tested, (
        "The following morphology tests need to be included in the mixed morphology tests:\n"
        f"{features_not_tested}"
    )

    return _dispatch_features(features, mode)


@pytest.mark.parametrize("feature_name, kwargs, expected", _morphology_features(mode="wout-subtrees"))
def test_morphology__morphology_features_wout_subtrees(feature_name, kwargs, expected, mixed_morph):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = get(feature_name, mixed_morph, use_subtrees=False, **kwargs)
        _assert_feature_equal(values, expected)


@pytest.mark.parametrize("feature_name, kwargs, expected", _morphology_features(mode="with-subtrees"))
def test_morphology__morphology_features_with_subtrees(
    feature_name, kwargs, expected, mixed_morph
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = get(feature_name, mixed_morph, use_subtrees=True, **kwargs)
        _assert_feature_equal(values, expected)


def _neurite_features():

    features = {
        "max_radial_distance": [
            # basal, AcD, apical
            {
                "kwargs": {"section_type": NeuriteType.all},
                "expected": [2.236068, 3.7416575, 2.236068],
            },
            {
                "kwargs": {"section_type": NeuriteType.all, "origin": np.array([0., 0., 0.])},
                "expected": [3.162277, 4.472135, 3.162277],
            },
            {
                "kwargs": {"section_type": NeuriteType.basal_dendrite},
                "expected": [2.236068, 3.3166249, 0.0],
            },
            {
                "kwargs": {"section_type": NeuriteType.basal_dendrite, "origin": np.array([0., 0., 0.])},
                "expected": [3.162277, 4.242640, 0.0]
            },
            {
                "kwargs": {"section_type": NeuriteType.axon},
                "expected": [0.      , 3.741657, 0.      ],
            },
            {
                "kwargs": {"section_type": NeuriteType.axon, "origin": np.array([0., 0., 0.])},
                "expected": [0.0, 4.472135, 0.0],
            }
        ],
        "volume_density": [
            {
                "kwargs": {"section_type": NeuriteType.all},
                "expected": [0.235619, 0.063784, 0.235619],
            },
            {
                "kwargs": {"section_type": NeuriteType.basal_dendrite},
                "expected": [0.235619, 0.255138, np.nan],
            },
            {
                "kwargs": {"section_type": NeuriteType.axon},
                "expected": [np.nan, 0.170092, np.nan],
            },
            {
                "kwargs": {"section_type": NeuriteType.apical_dendrite},
                "expected": [np.nan, np.nan, 0.2356194583819102],
            },
        ],
        "section_radial_distances": [
            {
                "kwargs": {"section_type": NeuriteType.all},
                "expected": [
                    [1.0, 2.0, 2.236068, 2.236068, 1.4142135],
                    [1.4142135, 3.1622777, 3.3166249, 3.3166249, 2.828427, 3.6055512, 3.6055512, 3.7416575, 3.7416575],
                    [1.0, 2.0, 2.236068, 2.236068, 1.4142135],
                ],
            },
            {
                "kwargs": {"section_type": NeuriteType.basal_dendrite},
                "expected": [
                    [1.0, 2.0, 2.236068, 2.236068, 1.4142135],
                    [1.414214, 3.162278, 3.316625, 3.316625],
                    [],
                ],
            },
            {
                "kwargs": {"section_type": NeuriteType.axon},
                "expected": [
                    [],
                    [2.828427, 3.605551, 3.605551, 3.741657, 3.741657],
                    [],
                ],
            },
            {
                "kwargs": {"section_type": NeuriteType.apical_dendrite},
                "expected": [
                    [],
                    [],
                    [1., 2., 2.236068, 2.236068, 1.414214],
                ],
            },
        ],
        "section_bif_radial_distances": [
            {
                "kwargs": {"section_type": NeuriteType.all},
                "expected": [
                    [1., 2.],
                    [1.414214, 3.162278, 2.828427, 3.605551],
                    [1., 2.],
                ],
            },
            {
                "kwargs": {"section_type": NeuriteType.basal_dendrite},
                "expected": [
                    [1., 2.],
                    [3.162278],
                    [],
                ],
            },
            {
                "kwargs": {"section_type": NeuriteType.axon},
                "expected": [
                    [],
                    [2.828427, 3.605551],
                    [],
                ],
            },
            {
                "kwargs": {"section_type": NeuriteType.apical_dendrite},
                "expected": [
                    [],
                    [],
                    [1., 2.],
                ],
            },
        ],
        "section_term_radial_distances": [
            {
                "kwargs": {"section_type": NeuriteType.all},
                "expected": [
                    [2.236068, 2.236068, 1.414214],
                    [3.316625, 3.316625, 3.605551, 3.741657, 3.741657],
                    [2.236068, 2.236068, 1.414214],
                ],
            },
            {
                "kwargs": {"section_type": NeuriteType.basal_dendrite},
                "expected": [
                    [2.236068, 2.236068, 1.414214],
                    [3.316625, 3.316625],
                    [],
                ],
            },
            {
                "kwargs": {"section_type": NeuriteType.axon},
                "expected": [
                    [],
                    [3.605551, 3.741657, 3.741657],
                    [],
                ],
            },
            {
                "kwargs": {"section_type": NeuriteType.apical_dendrite},
                "expected": [
                    [],
                    [],
                    [2.236068, 2.236068, 1.414214],
                ],
            },
        ],
        "segment_radial_distances": [
            {
                "kwargs": {"section_type": NeuriteType.all},
                "expected": [
                    [0.5     , 1.5     , 2.061553, 2.061553, 1.118034],
                    [0.707107, 2.236068, 3.201562, 3.201562, 2.12132 , 3.201562, 3.201562, 3.640055, 3.640055],
                    [0.5     , 1.5     , 2.061553, 2.061553, 1.118034],
                ],
            },
            {
                "kwargs": {"section_type": NeuriteType.basal_dendrite},
                "expected": [
                    [0.5     , 1.5     , 2.061553, 2.061553, 1.118034],
                    [0.707107, 2.236068, 3.201562, 3.201562],
                    [],
                ],
            },
            {
                "kwargs": {"section_type": NeuriteType.axon},
                "expected": [
                    [],
                    [2.12132 , 3.201562, 3.201562, 3.640055, 3.640055],
                    [],
                ],
            },
            {
                "kwargs": {"section_type": NeuriteType.apical_dendrite},
                "expected": [
                    [],
                    [],
                    [0.5     , 1.5     , 2.061553, 2.061553, 1.118034],
                ],
            },
        ]
    }

    # features that exist in both the neurite and morphology level, which indicates a different
    # implementation in each level
    features_not_tested = list(
        (set(_NEURITE_FEATURES)  & set(_MORPHOLOGY_FEATURES)) - features.keys()
    )

    assert not features_not_tested, (
        "The following morphology tests need to be included in the mixed neurite tests:\n\n" +
        "\n".join(sorted(features_not_tested)) + "\n"
    )

    return _dispatch_features(features)


@pytest.mark.parametrize("feature_name, kwargs, expected", _neurite_features())
def test_morphology__neurite_features(feature_name, kwargs, expected, mixed_morph):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = get(feature_name, mixed_morph.neurites, **kwargs)
        _assert_feature_equal(values, expected, per_neurite=True)
