import copy
import json
import pickle
import sys
import warnings
from pathlib import Path
import pytest
import neurom
import numpy as np
import pandas as pd
import numpy.testing as npt
from enum import Enum
from neurom import NeuriteType
from neurom.features import get
from neurom.core import Population
from neurom.features import _POPULATION_FEATURES, _MORPHOLOGY_FEATURES, _NEURITE_FEATURES
import collections.abc

from morphio import SectionType
from neurom.core.morphology import Section, iter_neurites, iter_sections
from neurom.core.types import tree_type_checker as is_type

import neurom.core.morphology
import neurom.features.neurite
import neurom.apps.morph_stats
from neurom.core.types import _ALL_SUBTYPE
from neurom.core.types import _SOMA_SUBTYPE
from neurom.core.types import SubtypeCollection
from neurom.core.types import MutableEnum
from neurom.core.types import NeuriteType
from neurom.exceptions import NeuroMError
from neurom.core.types import tree_type_checker as is_type


class TestSubtypeCollection:
    def test_repr(self):
        assert repr(SubtypeCollection(0)) == "(<SectionType.undefined: 0>,)"
        assert repr(SubtypeCollection(32)) == "(<SectionType.all: 32>,)"
        assert (
            repr(SubtypeCollection(3, 2, 1))
            == "(<SectionType.basal_dendrite: 3>, <SectionType.axon: 2>, <SectionType.soma: 1>)"
        )
        assert (
            repr(SubtypeCollection(NeuriteType.axon, NeuriteType.apical_dendrite))
            == "(<SectionType.axon: 2>, <SectionType.apical_dendrite: 4>)"
        )

    def test_str(self):
        assert str(SubtypeCollection(0)) == "SectionType.undefined"
        assert str(SubtypeCollection(32)) == "SectionType.all"
        assert (
            str(SubtypeCollection(3, 2, 1))
            == "SectionType.basal_dendrite-SectionType.axon-SectionType.soma"
        )
        assert (
            str(SubtypeCollection(NeuriteType.axon, NeuriteType.apical_dendrite))
            == "SectionType.axon-SectionType.apical_dendrite"
        )

    def test_int(self):
        assert int(SubtypeCollection(0)) == 0
        assert int(SubtypeCollection(32)) == 32
        assert int(SubtypeCollection(3, 2, 1)) == 30201
        assert int(SubtypeCollection(NeuriteType.axon, NeuriteType.apical_dendrite)) == 204

    def test_flatten(self):
        assert SubtypeCollection(NeuriteType.axon_carrying_dendrite) == (3, 2)
        assert SubtypeCollection(
            [[[3], [[NeuriteType.axon]], [[[SectionType(1), NeuriteType.axon_carrying_dendrite]]]]]
        ) == (3, 2, 1, 3, 2)

    def test_ctor(self):
        assert SubtypeCollection(1).subtypes == (SectionType.soma,)
        assert SubtypeCollection(1, 2).subtypes == (SectionType.soma, SectionType.axon)
        assert SubtypeCollection([1, 2]).subtypes == (SectionType.soma, SectionType.axon)
        assert SubtypeCollection(SubtypeCollection([1, 2])).subtypes == (
            SectionType.soma,
            SectionType.axon,
        )

        class TestEnum(Enum):
            a = 1
            b = 2

        assert SubtypeCollection(TestEnum.a, TestEnum.b).subtypes == (1, 2)

    def test_eq(self):
        assert SubtypeCollection(0) == 0
        assert SubtypeCollection(0) == SubtypeCollection(0)
        assert SubtypeCollection(32) == 32
        assert SubtypeCollection(32) == SubtypeCollection(32)
        assert SubtypeCollection(0) != 1
        assert SubtypeCollection(0) != SubtypeCollection(1)
        assert SubtypeCollection(3, 2, 1) == SubtypeCollection(3, 2, 1)
        assert SubtypeCollection(3, 2, 1) == SubtypeCollection(3)
        assert SubtypeCollection(3) == SubtypeCollection(3, 2, 1)
        assert SubtypeCollection(3, 2, 1) != SubtypeCollection(1, 2, 3)
        assert SubtypeCollection(3, 2, 1) == 3
        assert SubtypeCollection(3, 2, 1) == [3, 2, 1]
        assert SubtypeCollection(3, 2, 1) != [4, 5, 6]

        assert SubtypeCollection(NeuriteType.axon, NeuriteType.apical_dendrite) != 0
        assert SubtypeCollection(NeuriteType.axon, NeuriteType.apical_dendrite) == 2
        assert SubtypeCollection(NeuriteType.axon, NeuriteType.apical_dendrite) == NeuriteType.axon
        assert SubtypeCollection(NeuriteType.axon, NeuriteType.apical_dendrite) == [2, 4]
        assert SubtypeCollection(
            NeuriteType.axon, NeuriteType.apical_dendrite
        ) == SubtypeCollection(NeuriteType.axon, NeuriteType.apical_dendrite)

        assert SubtypeCollection(0) != "NOT A SUBTYPE"

    def test_raise(self):
        SubtypeCollection(32)

        with pytest.raises(
            NeuroMError,
            match=(
                r"A subtype containing the value 32 must contain only one element \(current "
                r"elements: \(<SectionType\.apical_dendrite: 4>, <SectionType\.all: 32>, "
                r"<SectionType\.custom6: 6>\)\)\."
            ),
        ):
            SubtypeCollection(4, 32, 6)

        with pytest.raises(
            ValueError,
            match="A SubtypeCollection object can not be empty",
        ):
            SubtypeCollection()

    def test_pickle(self):
        assert pickle.loads(pickle.dumps(SubtypeCollection(2))) == NeuriteType.axon
        assert pickle.loads(pickle.dumps(SubtypeCollection(1, 2, 3))) == [1, 2, 3]


def test_MutableEnum():
    with pytest.raises(
        TypeError, match=r"The class Foo must have a subtype given as the last parent"
    ):

        class Foo(MutableEnum):
            a = 1
            b = 2

    with pytest.raises(
        TypeError, match=r"The subtype of the Foo class must have a 'subtypes' attribute"
    ):

        class Foo(MutableEnum, int):
            a = 1
            b = 2


class TestNeuriteType:
    def test_ctor(self):
        assert NeuriteType.axon.name == "axon"
        assert NeuriteType.axon.subtypes == (SectionType(2),)
        assert NeuriteType("axon").name == "axon"
        assert NeuriteType("axon").subtypes == (SectionType(2),)
        assert NeuriteType(2).name == "axon"
        assert NeuriteType(2).subtypes == (SectionType(2),)

    def test_repr(self):
        assert repr(NeuriteType(0)) == "<NeuriteType.undefined: (<SectionType.undefined: 0>,)>"
        assert repr(NeuriteType(32)) == "<NeuriteType.all: (<SectionType.all: 32>,)>"
        assert repr(NeuriteType((3, 2))) == (
            "<NeuriteType.axon_carrying_dendrite: "
            "(<SectionType.basal_dendrite: 3>, <SectionType.axon: 2>)>"
        )

    def test_str(self):
        assert str(NeuriteType(0)) == "NeuriteType.undefined"
        assert str(NeuriteType(32)) == "NeuriteType.all"
        assert str(NeuriteType((3, 2))) == "NeuriteType.axon_carrying_dendrite"

    def test_eq(self):
        assert NeuriteType.axon == SectionType.axon
        assert NeuriteType.axon == 2
        assert NeuriteType.axon == SubtypeCollection(2)
        assert NeuriteType.axon == SubtypeCollection(SectionType.axon)
        assert NeuriteType.axon == SubtypeCollection(NeuriteType.axon)
        assert NeuriteType.axon == NeuriteType.axon
        assert NeuriteType.axon != 3
        assert NeuriteType.axon != SubtypeCollection(3)
        assert NeuriteType.axon != SubtypeCollection(SectionType.basal_dendrite)
        assert NeuriteType.axon != SubtypeCollection(NeuriteType.basal_dendrite)

        assert NeuriteType(2) == SectionType.axon
        assert NeuriteType(2) == 2
        assert NeuriteType(2) == SubtypeCollection(2)
        assert NeuriteType(2) == SubtypeCollection(SectionType.axon)
        assert NeuriteType(2) == SubtypeCollection(NeuriteType.axon)
        assert NeuriteType(2) == NeuriteType.axon
        assert NeuriteType(2) != 3
        assert NeuriteType(2) != SubtypeCollection(3)
        assert NeuriteType(2) != SubtypeCollection(SectionType.basal_dendrite)
        assert NeuriteType(2) != SubtypeCollection(NeuriteType.basal_dendrite)

        assert NeuriteType([3, 2]) == SubtypeCollection(3, 2)
        assert NeuriteType([3, 2]) == NeuriteType([3, 2])
        assert NeuriteType([3, 2]) == SubtypeCollection(SectionType.axon)
        assert NeuriteType([3, 2]) == NeuriteType(SectionType.axon)
        assert NeuriteType([3, 2]) == SubtypeCollection(
            NeuriteType.basal_dendrite, NeuriteType.axon
        )
        assert NeuriteType([3, 2]) != SubtypeCollection(
            NeuriteType.axon, NeuriteType.apical_dendrite
        )
        assert NeuriteType([3, 2]) != [NeuriteType.axon, NeuriteType.apical_dendrite]
        assert NeuriteType([3, 2]) == NeuriteType.axon
        assert NeuriteType([3, 2]) == NeuriteType.basal_dendrite
        assert NeuriteType([3, 2]) != NeuriteType.apical_dendrite
        assert NeuriteType([3, 2]) != SubtypeCollection(4, 3, 2)
        assert NeuriteType([3, 2]) != [4, 3, 2]
        assert NeuriteType([3, 2]) != SubtypeCollection(999)

        assert NeuriteType.axon_carrying_dendrite == SectionType.axon
        assert NeuriteType.axon_carrying_dendrite == 2
        assert NeuriteType.axon_carrying_dendrite == SubtypeCollection(2)
        assert NeuriteType.axon_carrying_dendrite == SubtypeCollection(SectionType.axon)
        assert NeuriteType.axon_carrying_dendrite == SubtypeCollection(NeuriteType.axon)
        assert NeuriteType.axon_carrying_dendrite == NeuriteType.axon
        assert NeuriteType.axon_carrying_dendrite == NeuriteType(2)
        assert NeuriteType.axon_carrying_dendrite == (SectionType.basal_dendrite, SectionType.axon)
        assert NeuriteType.axon_carrying_dendrite == SubtypeCollection(
            SectionType.basal_dendrite, SectionType.axon
        )
        assert NeuriteType.axon_carrying_dendrite == NeuriteType(
            [SectionType.basal_dendrite, SectionType.axon]
        )
        assert NeuriteType.axon_carrying_dendrite == 3
        assert NeuriteType.axon_carrying_dendrite == SubtypeCollection(3)
        assert NeuriteType.axon_carrying_dendrite == SubtypeCollection(SectionType.basal_dendrite)
        assert NeuriteType.axon_carrying_dendrite == SubtypeCollection(NeuriteType.basal_dendrite)
        assert NeuriteType.axon_carrying_dendrite == NeuriteType.basal_dendrite
        assert NeuriteType.axon_carrying_dendrite == NeuriteType(3)
        assert NeuriteType.axon_carrying_dendrite != 4
        assert NeuriteType.axon_carrying_dendrite != SubtypeCollection(4)
        assert NeuriteType.axon_carrying_dendrite != NeuriteType(4)
        assert NeuriteType.axon_carrying_dendrite != SubtypeCollection(SectionType.apical_dendrite)
        assert NeuriteType.axon_carrying_dendrite != SubtypeCollection(NeuriteType.apical_dendrite)

    def test_raise(self):
        NeuriteType("all")
        NeuriteType(NeuriteType.all)
        NeuriteType((3, 2))
        with pytest.raises(ValueError, match="None is not a valid registered NeuriteType"):
            NeuriteType(None)
        with pytest.raises(
            ValueError, match="{'WRONG TYPE': 999} is not a valid registered NeuriteType"
        ):
            NeuriteType({"WRONG TYPE": 999})
        with pytest.raises(ValueError, match="UNKNOWN VALUE is not a valid registered NeuriteType"):
            NeuriteType("UNKNOWN VALUE")
        with pytest.raises(ValueError, match=r"\[2, 3, 4\] is not a valid registered NeuriteType"):
            NeuriteType([2, 3, 4])

        with pytest.raises(ValueError, match="20304 is not a valid registered NeuriteType"):
            NeuriteType(20304)

        with pytest.raises(
            ValueError, match="The NeuriteType class constructor accepts only 1 argument."
        ):
            NeuriteType(1, 2, 3)

    def test_pickle(self):
        assert pickle.loads(pickle.dumps(NeuriteType(2))) == NeuriteType.axon
        assert pickle.loads(pickle.dumps(NeuriteType.axon)) == NeuriteType.axon

    @pytest.fixture
    def reset_NeuriteType(self):
        current_dict = dict(NeuriteType.__dict__.items())
        # current_value2member_map_ = copy.deepcopy(NeuriteType._value2member_map_)
        # current_member_map_ = copy.deepcopy(NeuriteType._member_map_)
        # current_member_names_ = copy.deepcopy(NeuriteType._member_names_)
        yield
        for k, v in current_dict.items():
            setattr(NeuriteType, k, v)
        for k in list(NeuriteType.__dict__.keys()):
            if k not in current_dict:
                delattr(NeuriteType, k)
        # NeuriteType._value2member_map_ = current_value2member_map_
        # NeuriteType._member_map_ = current_member_map_
        # NeuriteType._member_names_ = current_member_names_

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(99, id="Simple scalar value"),
            pytest.param([SectionType.axon, SectionType.soma], id="Composite value"),
        ],
    )
    def test_register_unregister(self, value, reset_NeuriteType):
        obj = NeuriteType.register("new_type", value)
        assert NeuriteType(value) == obj
        assert NeuriteType(value).name == "new_type"
        assert NeuriteType(value) == SubtypeCollection(value)
        # assert NeuriteType(value).value == SubtypeCollection(value)
        assert getattr(NeuriteType, "new_type") == obj
        # assert NeuriteType["new_type"] == obj

        with pytest.raises(ValueError):
            # Try to register a new type with already existing value
            NeuriteType.register("other_new_type", value)

        with pytest.raises(ValueError):
            # Try to register a new type with already existing name
            NeuriteType.register("axon", 88)

        NeuriteType.unregister("new_type")

        with pytest.raises(ValueError):
            # Try to unregister an unregistered value
            NeuriteType.unregister("UNKNOWN VALUE")

        with pytest.raises(ValueError):
            # Try to unregister an existing attribute that is not a registered value
            NeuriteType.unregister("name")

        with pytest.raises(ValueError):
            # Try to get unregistered value
            NeuriteType(value)


DATA_DIR = Path(__file__).parent / "data/mixed"


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
                                                     |     S18
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
        reader="swc",
    )


@pytest.fixture
def population(mixed_morph):
    return Population([mixed_morph, mixed_morph])


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
        reader="swc",
    )


def test_heterogeneous_neurites(mixed_morph):
    assert not mixed_morph.neurites[0].is_heterogeneous()
    assert mixed_morph.neurites[1].is_heterogeneous()
    assert not mixed_morph.neurites[2].is_heterogeneous()


def test_iter_sections(mixed_morph):
    # Test homogenous trees
    mixed_morph.process_subtrees = False
    # # Iterate with ipreorder iterator
    assert [i.id for i in iter_sections(mixed_morph)] == list(range(19))
    assert [
        i.id for i in iter_sections(mixed_morph, neurite_filter=is_type(NeuriteType.all))
    ] == list(range(19))
    assert [
        i.id for i in iter_sections(mixed_morph, neurite_filter=is_type(NeuriteType.axon))
    ] == []
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            neurite_filter=is_type(NeuriteType.axon),
            section_filter=is_type(NeuriteType.all),
        )
    ] == []

    # # Iterate with ibifurcation_point iterator
    assert [
        i.id for i in iter_sections(mixed_morph, iterator_type=Section.ibifurcation_point)
    ] == [0, 1, 5, 6, 9, 11, 14, 15]  # fmt: skip
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            iterator_type=Section.ibifurcation_point,
            neurite_filter=is_type(NeuriteType.all),
        )
    ] == [0, 1, 5, 6, 9, 11, 14, 15]
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            iterator_type=Section.ibifurcation_point,
            neurite_filter=is_type(NeuriteType.axon),
        )
    ] == []
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            iterator_type=Section.ibifurcation_point,
            neurite_filter=is_type(NeuriteType.axon),
            section_filter=is_type(NeuriteType.all),
        )
    ] == []

    # Test heterogenous trees
    mixed_morph.process_subtrees = True
    # # Iterate with ipreorder iterator
    assert [i.id for i in iter_sections(mixed_morph)] == list(range(19))
    assert [
        i.id for i in iter_sections(mixed_morph, neurite_filter=is_type(NeuriteType.all))
    ] == list(range(19))
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            neurite_filter=is_type(NeuriteType.axon),
            section_filter=is_type(NeuriteType.all),
        )
    ] == [5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            neurite_filter=is_type(NeuriteType.axon),
            section_filter=is_type(NeuriteType.axon),
        )
    ] == [9, 10, 11, 12, 13]
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            neurite_filter=is_type(NeuriteType.axon),
            section_filter=is_type(NeuriteType.basal_dendrite),
        )
    ] == [5, 6, 7, 8]

    # # Iterate with ibifurcation_point iterator
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            iterator_type=Section.ibifurcation_point,
        )
    ] == [0, 1, 5, 6, 9, 11, 14, 15]
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            iterator_type=Section.ibifurcation_point,
            neurite_filter=is_type(NeuriteType.all),
        )
    ] == [0, 1, 5, 6, 9, 11, 14, 15]
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            iterator_type=Section.ibifurcation_point,
            neurite_filter=is_type(NeuriteType.axon),
        )
    ] == [5, 6, 9, 11]
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            iterator_type=Section.ibifurcation_point,
            neurite_filter=is_type(NeuriteType.axon),
            section_filter=is_type(NeuriteType.all),
        )
    ] == [5, 6, 9, 11]
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            iterator_type=Section.ibifurcation_point,
            neurite_filter=is_type(NeuriteType.axon),
            section_filter=is_type(NeuriteType.axon),
        )
    ] == [9, 11]
    assert [
        i.id
        for i in iter_sections(
            mixed_morph,
            iterator_type=Section.ibifurcation_point,
            neurite_filter=is_type(NeuriteType.axon),
            section_filter=is_type(NeuriteType.basal_dendrite),
        )
    ] == [5, 6]


def test_is_homogeneous_point(mixed_morph):
    heterogeneous_neurite = mixed_morph.neurites[1]

    sections = list(heterogeneous_neurite.iter_sections())

    # first section has one axon and one basal children
    assert not sections[0].is_homogeneous_point()

    # second section is pure basal
    assert sections[1].is_homogeneous_point()


def test_subtypes(mixed_morph):
    homogeneous_neurite = mixed_morph.neurites[0]
    heterogeneous_neurite = mixed_morph.neurites[1]

    assert homogeneous_neurite.subtree_types == [NeuriteType.basal_dendrite, NeuriteType.axon]
    assert homogeneous_neurite.type == NeuriteType.basal_dendrite

    assert heterogeneous_neurite.subtree_types == [NeuriteType.basal_dendrite, NeuriteType.axon]
    assert heterogeneous_neurite.type == NeuriteType.axon_carrying_dendrite


def test_number_of_sections(mixed_morph, population):
    # Count number of sections with process_subtrees == False
    # # Population
    # # In this case only the neurite_type argument is considered but the section_type argument is ignored.
    assert get('number_of_sections', population) == [19, 19]
    assert get('number_of_sections', population, neurite_type=NeuriteType.all) == [19, 19]
    assert get('number_of_sections', population, neurite_type=NeuriteType.axon) == [0, 0]
    assert get('number_of_sections', population, neurite_type=NeuriteType.apical_dendrite) == [5, 5]
    assert get('number_of_sections', population, neurite_type=NeuriteType.basal_dendrite) == [
        14,
        14,
    ]
    with pytest.raises(NeuroMError, match='Can not apply "section_type" arg to a Population'):
        get('number_of_sections', population, section_type=NeuriteType.soma)

    # # Morphology
    # # In this case only the neurite_type argument is considered but the section_type argument is ignored.
    assert get('number_of_sections', mixed_morph) == 19
    assert get('number_of_sections', mixed_morph, neurite_type=NeuriteType.all) == 19
    assert get('number_of_sections', mixed_morph, neurite_type=NeuriteType.axon) == 0
    assert get('number_of_sections', mixed_morph, neurite_type=NeuriteType.apical_dendrite) == 5
    assert get('number_of_sections', mixed_morph, neurite_type=NeuriteType.basal_dendrite) == 14
    assert get('number_of_sections', mixed_morph, neurite_type=NeuriteType.soma) == 0
    assert get('number_of_sections', mixed_morph, neurite_type=NeuriteType.undefined) == 0
    with pytest.raises(NeuroMError, match='Can not apply "section_type" arg to a Morphology'):
        get('number_of_sections', mixed_morph, section_type=NeuriteType.soma)

    # # List of neurites
    # # In this case the process_subtrees flag is ignored. So only the section with the proper
    # # section type are considered but all bifurcation points are kept, even heterogeneous ones.
    assert get('number_of_sections', mixed_morph.neurites) == [5, 9, 5]
    assert get('number_of_sections', mixed_morph.neurites, section_type=NeuriteType.all) == [
        5,
        9,
        5,
    ]
    assert get('number_of_sections', mixed_morph.neurites, section_type=NeuriteType.axon) == [
        0,
        5,
        0,
    ]
    assert get(
        'number_of_sections', mixed_morph.neurites, section_type=NeuriteType.apical_dendrite
    ) == [0, 0, 5]
    assert get(
        'number_of_sections', mixed_morph.neurites, section_type=NeuriteType.basal_dendrite
    ) == [5, 4, 0]
    with pytest.raises(
        NeuroMError, match='Can not apply "neurite_type" arg to a Neurite with a neurite feature'
    ):
        assert get('number_of_sections', mixed_morph.neurites, neurite_type=NeuriteType.all)

    # # One neurite (in this case the process_subtrees flag is ignored)
    assert get('number_of_sections', mixed_morph.neurites[1]) == 9
    assert get('number_of_sections', mixed_morph.neurites[1], section_type=NeuriteType.all) == 9
    assert get('number_of_sections', mixed_morph.neurites[1], section_type=NeuriteType.axon) == 5
    assert (
        get('number_of_sections', mixed_morph.neurites[1], section_type=NeuriteType.apical_dendrite)
        == 0
    )
    assert (
        get('number_of_sections', mixed_morph.neurites[1], section_type=NeuriteType.basal_dendrite)
        == 4
    )
    with pytest.raises(
        NeuroMError, match='Can not apply "neurite_type" arg to a Neurite with a neurite feature'
    ):
        assert get('number_of_sections', mixed_morph.neurites[1], neurite_type=NeuriteType.all)

    # Count number of sections with process_subtrees == True
    population.process_subtrees = True
    for i in population:
        assert i.process_subtrees is True
    mixed_morph.process_subtrees = True
    assert mixed_morph.process_subtrees is True

    # # Population
    # # In this case only the neurite_type argument is considered but the section_type argument is ignored.
    assert get('number_of_sections', population) == [19, 19]
    assert get('number_of_sections', population, neurite_type=NeuriteType.all) == [19, 19]
    assert get('number_of_sections', population, neurite_type=NeuriteType.axon) == [5, 5]
    assert get('number_of_sections', population, neurite_type=NeuriteType.apical_dendrite) == [5, 5]
    assert get('number_of_sections', population, neurite_type=NeuriteType.basal_dendrite) == [
        9,
        9,
    ]  # This is weird: we skip bifurcation points but we still count 2 sections around heterogeneous bifurcation points
    with pytest.raises(NeuroMError, match='Can not apply "section_type" arg to a Population'):
        get('number_of_sections', population, section_type=NeuriteType.soma)

    # # Morphology
    # # In this case only the neurite_type argument is considered but the section_type argument is ignored.
    assert get('number_of_sections', mixed_morph) == 19
    assert get('number_of_sections', mixed_morph, neurite_type=NeuriteType.all) == 19
    assert get('number_of_sections', mixed_morph, neurite_type=NeuriteType.axon) == 5
    assert get('number_of_sections', mixed_morph, neurite_type=NeuriteType.apical_dendrite) == 5
    assert get('number_of_sections', mixed_morph, neurite_type=NeuriteType.basal_dendrite) == 9
    assert get('number_of_sections', mixed_morph, neurite_type=NeuriteType.soma) == 0
    assert get('number_of_sections', mixed_morph, neurite_type=NeuriteType.undefined) == 0
    with pytest.raises(NeuroMError, match='Can not apply "section_type" arg to a Morphology'):
        get('number_of_sections', mixed_morph, section_type=NeuriteType.soma)

    # # List of neurites
    # # In this case the process_subtrees flag is ignored. So only the section with the proper
    # # section type are considered but all bifurcation points are kept, even heterogeneous ones.
    assert get('number_of_sections', mixed_morph.neurites) == [5, 9, 5]
    assert get('number_of_sections', mixed_morph.neurites, section_type=NeuriteType.all) == [
        5,
        9,
        5,
    ]
    assert get('number_of_sections', mixed_morph.neurites, section_type=NeuriteType.axon) == [
        0,
        5,
        0,
    ]
    assert get(
        'number_of_sections', mixed_morph.neurites, section_type=NeuriteType.apical_dendrite
    ) == [0, 0, 5]
    assert get(
        'number_of_sections', mixed_morph.neurites, section_type=NeuriteType.basal_dendrite
    ) == [5, 4, 0]
    with pytest.raises(
        NeuroMError, match='Can not apply "neurite_type" arg to a Neurite with a neurite feature'
    ):
        assert get('number_of_sections', mixed_morph.neurites, neurite_type=NeuriteType.all)

    # # One neurite (in this case the process_subtrees flag is ignored)
    assert get('number_of_sections', mixed_morph.neurites[1]) == 9
    assert get('number_of_sections', mixed_morph.neurites[1], section_type=NeuriteType.all) == 9
    assert get('number_of_sections', mixed_morph.neurites[1], section_type=NeuriteType.axon) == 5
    assert (
        get('number_of_sections', mixed_morph.neurites[1], section_type=NeuriteType.apical_dendrite)
        == 0
    )
    assert (
        get('number_of_sections', mixed_morph.neurites[1], section_type=NeuriteType.basal_dendrite)
        == 4
    )
    with pytest.raises(
        NeuroMError, match='Can not apply "neurite_type" arg to a Neurite with a neurite feature'
    ):
        assert get('number_of_sections', mixed_morph.neurites[1], neurite_type=NeuriteType.all)


def test_iter_neurites__heterogeneous(mixed_morph):
    mixed_morph.process_subtrees = True

    neurites = list(iter_neurites(mixed_morph))

    assert len(neurites) == 3
    assert neurites[0].type == NeuriteType.basal_dendrite
    assert neurites[1].type == NeuriteType.basal_dendrite
    assert neurites[2].type == NeuriteType.apical_dendrite


def test_iter_neurites__homogeneous(mixed_morph):
    mixed_morph.process_subtrees = False

    neurites = list(iter_neurites(mixed_morph))

    assert len(neurites) == 3
    assert neurites[0].type == NeuriteType.basal_dendrite
    assert neurites[1].type == NeuriteType.axon_carrying_dendrite
    assert neurites[2].type == NeuriteType.apical_dendrite


def test_core_iter_sections__heterogeneous(mixed_morph):
    mixed_morph.process_subtrees = True

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
    assert_sections(
        axon_on_basal,
        (NeuriteType.axon, NeuriteType.basal_dendrite),
        [5, 6, 7, 8, 9, 10, 11, 12, 13],
    )

    assert_sections(apical, NeuriteType.all, [14, 15, 16, 17, 18])
    assert_sections(apical, NeuriteType.apical_dendrite, [14, 15, 16, 17, 18])


def test_features_neurite_map_sections__heterogeneous(mixed_morph):
    mixed_morph.process_subtrees = True

    def assert_sections(neurite, section_type, iterator_type, expected_section_ids):
        function = lambda section: section.id
        section_ids = neurom.features.neurite._map_sections(
            function, neurite, iterator_type=iterator_type, section_type=section_type
        )
        assert section_ids == expected_section_ids

    basal, axon_on_basal, apical = mixed_morph.neurites

    # homogeneous tree, no difference between all and basal_dendrite types.
    assert_sections(
        basal,
        NeuriteType.all,
        neurom.core.morphology.Section.ibifurcation_point,
        [0, 1],
    )
    assert_sections(
        basal,
        NeuriteType.basal_dendrite,
        neurom.core.morphology.Section.ibifurcation_point,
        [0, 1],
    )
    # heterogeneous tree, forks cannot be heterogeneous if a type other than all is specified
    # Section with id 5 is the transition section, which has a basal and axon children sections
    assert_sections(
        axon_on_basal,
        NeuriteType.all,
        neurom.core.morphology.Section.ibifurcation_point,
        [5, 6, 9, 11],
    )
    assert_sections(
        axon_on_basal,
        NeuriteType.basal_dendrite,
        neurom.core.morphology.Section.ibifurcation_point,
        [6],
    )
    assert_sections(
        axon_on_basal,
        NeuriteType.axon,
        neurom.core.morphology.Section.ibifurcation_point,
        [9, 11],
    )
    # homogeneous tree, no difference between all and basal_dendrite types.
    assert_sections(
        apical,
        NeuriteType.all,
        neurom.core.morphology.Section.ibifurcation_point,
        [14, 15],
    )
    assert_sections(
        apical,
        NeuriteType.apical_dendrite,
        neurom.core.morphology.Section.ibifurcation_point,
        [14, 15],
    )
    # with composite type the whole heterogeneous tree is kept
    assert_sections(
        axon_on_basal,
        NeuriteType.axon_carrying_dendrite,
        neurom.core.morphology.Section.ibifurcation_point,
        [5, 6, 9, 11],
    )


def test_features_neurite_map_sections(mixed_morph):
    mixed_morph.process_subtrees = False
    acd = mixed_morph.neurites[1]

    def count(iterator_type, section_type):
        return sum(
            neurom.features.neurite._map_sections(
                fun=lambda s: 1,
                neurite=acd,
                iterator_type=iterator_type,
                section_type=section_type,
            )
        )

    res = count(Section.ipreorder, NeuriteType.all)
    assert res == 9

    res = count(Section.ipreorder, NeuriteType.axon)
    assert res == 5

    res = count(Section.ipreorder, NeuriteType.basal_dendrite)
    assert res == 4

    res = count(Section.ipreorder, NeuriteType.axon_carrying_dendrite)
    assert res == 9

    res = count(Section.ibifurcation_point, NeuriteType.all)
    assert res == 4

    res = count(Section.ibifurcation_point, NeuriteType.basal_dendrite)
    assert res == 1

    res = count(Section.ibifurcation_point, NeuriteType.axon)
    assert res == 2

    res = count(Section.ibifurcation_point, NeuriteType.axon_carrying_dendrite)
    assert res == 4

    res = count(Section.ibifurcation_point, (NeuriteType.axon, NeuriteType.basal_dendrite))
    assert res == 4


def _assert_stats_equal(actual_dict, expected_dict):
    assert actual_dict.keys() == expected_dict.keys()
    for key, value in actual_dict.items():
        expected_value = expected_dict[key]
        if value is None or expected_value is None:
            assert expected_value is value
        else:
            npt.assert_almost_equal(value, expected_value, decimal=3, err_msg=f"\nKey: {key}")


@pytest.fixture
def stats_cfg():
    return {
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
            'section_strahler_orders': ['min'],
        },
        'morphology': {
            'soma_surface_area': ['mean'],
            'soma_radius': ['max'],
            'max_radial_distance': ['mean'],
            'number_of_sections_per_neurite': ['median'],
            'total_length_per_neurite': ['mean'],
            'total_area_per_neurite': ['mean'],
            'total_volume_per_neurite': ['mean'],
            'number_of_neurites': ['median'],
        },
        'neurite_type': ['AXON', 'BASAL_DENDRITE', 'APICAL_DENDRITE'],
    }


def test_mixed__extract_stats__homogeneous(stats_cfg, mixed_morph):
    mixed_morph.process_subtrees = False
    res = neurom.apps.morph_stats.extract_stats(mixed_morph, stats_cfg)

    expected = {
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
        'min_total_length': 0,
    }

    _assert_stats_equal(res["axon"], expected)

    res_df = neurom.apps.morph_stats.extract_dataframe(mixed_morph, stats_cfg)

    # get axon column and tranform it to look like the expected values above
    values = res_df.loc[pd.IndexSlice[:, "axon"]].iloc[0, :].to_dict()
    _assert_stats_equal(values, expected)


def test_mixed__extract_stats__heterogeneous(stats_cfg, mixed_morph):
    mixed_morph.process_subtrees = True
    res = neurom.apps.morph_stats.extract_stats(mixed_morph, stats_cfg)

    expected = {
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
        'mean_section_path_distances': 4.028427076339722,
        'mean_section_radial_distances': 4.207625,
        'mean_section_term_branch_orders': 2.6666666666666665,
        'mean_section_term_lengths': 1.0,
        'mean_section_term_radial_distances': 4.396645,
        'mean_section_tortuosity': 1.0,
        'mean_sibling_ratios': 1.0,
        'mean_terminal_path_lengths': 4.495093743006389,
        'median_diameter_power_relations': 2.0,
        'median_number_of_leaves': 3,
        'median_section_taper_rates': 8.6268466e-17,
        'median_total_volume': 0.17009254152367845,
        'min_number_of_sections': 5,
        'min_section_strahler_orders': 1,
        'min_section_volumes': 0.03141592778425469,
        'min_total_length': 5.414213538169861,
    }

    _assert_stats_equal(res["axon"], expected)

    res_df = neurom.apps.morph_stats.extract_dataframe(mixed_morph, stats_cfg)

    # get axon column and tranform it to look like the expected values above
    values = res_df.loc[pd.IndexSlice[:, "axon"]].iloc[0, :].to_dict()
    _assert_stats_equal(values, expected)


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
    features = json.loads(Path(DATA_DIR / "expected_population_features.json").read_bytes())

    features_not_tested = list(set(_POPULATION_FEATURES) - set(features.keys()))

    assert not features_not_tested, (
        "The following morphology tests need to be included in the tests:\n\n"
        + "\n".join(sorted(features_not_tested))
        + "\n"
    )

    return _dispatch_features(features, mode)


@pytest.mark.parametrize(
    "feature_name, kwargs, expected", _population_features(mode="wout-subtrees")
)
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_population__population_features_wout_subtrees(feature_name, kwargs, expected, population):
    population.process_subtrees = False
    values = get(feature_name, population, **kwargs)
    _assert_feature_equal(values, expected)


@pytest.mark.parametrize(
    "feature_name, kwargs, expected", _population_features(mode="with-subtrees")
)
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_population__population_features_with_subtrees(feature_name, kwargs, expected, population):
    population.process_subtrees = True
    values = get(feature_name, population, **kwargs)
    _assert_feature_equal(values, expected)


def _morphology_features(mode):
    features = json.loads(Path(DATA_DIR / "expected_morphology_features.json").read_bytes())

    features_not_tested = (set(_MORPHOLOGY_FEATURES) | set(_NEURITE_FEATURES)) - set(
        features.keys()
    )

    assert not features_not_tested, (
        "The following morphology tests need to be included in the mixed morphology tests:\n"
        f"{features_not_tested}"
    )

    return _dispatch_features(features, mode)


@pytest.mark.parametrize(
    "feature_name, kwargs, expected", _morphology_features(mode="wout-subtrees")
)
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_morphology__morphology_features_wout_subtrees(feature_name, kwargs, expected, mixed_morph):
    mixed_morph.process_subtrees = False
    values = get(feature_name, mixed_morph, **kwargs)
    _assert_feature_equal(values, expected)


@pytest.mark.parametrize(
    "feature_name, kwargs, expected", _morphology_features(mode="with-subtrees")
)
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_morphology__morphology_features_with_subtrees(feature_name, kwargs, expected, mixed_morph):
    mixed_morph.process_subtrees = True
    values = get(feature_name, mixed_morph, **kwargs)
    _assert_feature_equal(values, expected)


def _neurite_features():
    features = json.loads(Path(DATA_DIR / "expected_neurite_features.json").read_bytes())

    # features that exist in both the neurite and morphology level, which indicates a different
    # implementation in each level
    features_not_tested = list(
        (set(_NEURITE_FEATURES) & set(_MORPHOLOGY_FEATURES)) - features.keys()
    )

    assert not features_not_tested, (
        "The following morphology tests need to be included in the mixed neurite tests:\n\n"
        + "\n".join(sorted(features_not_tested))
        + "\n"
    )

    return _dispatch_features(features)


@pytest.mark.parametrize("feature_name, kwargs, expected", _neurite_features())
def test_morphology__neurite_features(feature_name, kwargs, expected, mixed_morph):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = get(feature_name, mixed_morph.neurites, **kwargs)
        _assert_feature_equal(values, expected, per_neurite=True)


def test_sholl_crossings(mixed_morph):
    mixed_morph.process_subtrees = True
    center = mixed_morph.soma.center
    radii = []
    assert list(get("sholl_crossings", mixed_morph, center=center, radii=radii)) == []
    assert list(get("sholl_crossings", mixed_morph, radii=radii)) == []
    assert list(get("sholl_crossings", mixed_morph)) == [0]

    radii = [1.0]
    assert list(get("sholl_crossings", mixed_morph, center=center, radii=radii)) == [3]

    radii = [1.0, 4.0]
    assert list(get("sholl_crossings", mixed_morph, center=center, radii=radii)) == [3, 3]

    radii = [1.0, 4.0, 5.0]
    assert list(get("sholl_crossings", mixed_morph, center=center, radii=radii)) == [3, 3, 0]

    radii = [1.0, 4.0, 5.0, 10]
    assert list(
        get(
            "sholl_crossings", mixed_morph, neurite_type=NeuriteType.all, center=center, radii=radii
        )
    ) == [3, 3, 0, 0]
    assert list(
        get(
            "sholl_crossings",
            mixed_morph,
            neurite_type=NeuriteType.basal_dendrite,
            center=center,
            radii=radii,
        )
    ) == [2, 1, 0, 0]
    assert list(
        get(
            "sholl_crossings",
            mixed_morph,
            neurite_type=NeuriteType.apical_dendrite,
            center=center,
            radii=radii,
        )
    ) == [1, 0, 0, 0]
    assert list(
        get(
            "sholl_crossings",
            mixed_morph,
            neurite_type=NeuriteType.axon,
            center=center,
            radii=radii,
        )
    ) == [0, 2, 0, 0]


def test_sholl_frequency(mixed_morph):
    mixed_morph.process_subtrees = True
    assert list(get("sholl_frequency", mixed_morph)) == [0]
    assert list(get("sholl_frequency", mixed_morph, step_size=3)) == [0, 2]
    assert list(get("sholl_frequency", mixed_morph, bins=[1, 3, 5])) == [3, 8, 0]

    assert list(get("sholl_frequency", mixed_morph, neurite_type=NeuriteType.basal_dendrite)) == [0]
    assert list(
        get("sholl_frequency", mixed_morph, neurite_type=NeuriteType.basal_dendrite, step_size=3)
    ) == [0, 1]
    assert list(
        get("sholl_frequency", mixed_morph, neurite_type=NeuriteType.basal_dendrite, bins=[1, 3, 5])
    ) == [2, 4, 0]

    assert list(get("sholl_frequency", mixed_morph, neurite_type=NeuriteType.axon)) == [0]
    assert list(
        get("sholl_frequency", mixed_morph, neurite_type=NeuriteType.axon, step_size=3)
    ) == [0, 1]
    assert list(
        get("sholl_frequency", mixed_morph, neurite_type=NeuriteType.axon, bins=[1, 3, 5])
    ) == [0, 1, 0]


def test_sholl_frequency_pop(mixed_morph):
    pop = Population([mixed_morph, mixed_morph])
    pop.process_subtrees = True
    assert list(get("sholl_frequency", pop)) == [0]
    assert list(get("sholl_frequency", pop, step_size=3)) == [0, 4]
    assert list(get("sholl_frequency", pop, bins=[1, 3, 5])) == [6, 16, 0]

    assert list(get("sholl_frequency", pop, neurite_type=NeuriteType.basal_dendrite)) == [0]
    assert list(
        get("sholl_frequency", pop, neurite_type=NeuriteType.basal_dendrite, step_size=3)
    ) == [0, 2]
    assert list(
        get("sholl_frequency", pop, neurite_type=NeuriteType.basal_dendrite, bins=[1, 3, 5])
    ) == [4, 8, 0]

    assert list(get("sholl_frequency", pop, neurite_type=NeuriteType.axon)) == [0]
    assert list(get("sholl_frequency", pop, neurite_type=NeuriteType.axon, step_size=3)) == [0, 2]
    assert list(get("sholl_frequency", pop, neurite_type=NeuriteType.axon, bins=[1, 3, 5])) == [
        0,
        2,
        0,
    ]
