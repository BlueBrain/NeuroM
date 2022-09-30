"""Miscelaneous tests of features."""
from pathlib import Path
from itertools import chain

import numpy as np
import pytest
from numpy import testing as npt

import neurom as nm
from neurom import features


@pytest.fixture
def DATA_PATH():
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def SWC_PATH(DATA_PATH):
    return DATA_PATH / "swc"


@pytest.fixture
def MORPHOLOGY(SWC_PATH):
    return nm.load_morphology(SWC_PATH / "test_morph.swc")


@pytest.fixture
def NEURITE(MORPHOLOGY):
    return MORPHOLOGY.neurites[0]


@pytest.fixture
def SECTION(NEURITE):
    return NEURITE.sections[0]


@pytest.fixture
def NRN_FILES(DATA_PATH):
    return [
        DATA_PATH / "h5/v1" / f for f in ("Neuron.h5", "Neuron_2_branch.h5", "bio_neuron-001.h5")
    ]


@pytest.fixture
def POP(NRN_FILES):
    return nm.load_morphologies(NRN_FILES)


def _check_nested_type(data):
    """Check that the given data contains only built-in types.

    The data should either be an int or float, or a list or tuple of ints or floats.
    """
    if isinstance(data, (list, tuple)):
        for i in data:
            _check_nested_type(i)
    else:
        assert isinstance(data, (int, float))


class TestFeatureTypes:
    """Test that all features return raw Python types."""

    @pytest.mark.parametrize(
        "feature_name",
        [
            pytest.param(name, id=f"Test type of {name} neurite feature")
            for name in features._NEURITE_FEATURES
        ],
    )
    def test_neurite_feature_types(self, feature_name, NEURITE):
        """Test neurite features."""
        res = features._NEURITE_FEATURES.get(feature_name)(NEURITE)
        _check_nested_type(res)

    @pytest.mark.parametrize(
        "feature_name",
        [
            pytest.param(name, id=f"Test type of {name} morphology feature")
            for name in features._MORPHOLOGY_FEATURES
        ],
    )
    def test_morphology_feature_types(self, feature_name, MORPHOLOGY):
        """Test morphology features."""
        res = features._MORPHOLOGY_FEATURES.get(feature_name)(MORPHOLOGY)
        _check_nested_type(res)

    @pytest.mark.parametrize(
        "feature_name",
        [
            pytest.param(name, id=f"Test type of {name} population feature")
            for name in features._POPULATION_FEATURES
        ],
    )
    def test_population_feature_types(self, feature_name, POP):
        """Test population features."""
        res = features._POPULATION_FEATURES.get(feature_name)(POP)
        _check_nested_type(res)
