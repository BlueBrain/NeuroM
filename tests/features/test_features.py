"""Miscelaneous tests of features."""
from pathlib import Path
from itertools import chain

import numpy as np
import pytest
from numpy import testing as npt

import neurom as nm
from neurom import features


def _check_nested_type(data):
    """Check that the given data contains only built-in types.

    The data should either be an int or float, or a list or tuple of ints or floats.
    """
    if isinstance(data, (list, tuple)):
        for i in data:
            _check_nested_type(i)
    else:
        assert isinstance(data, (int, float))


@pytest.mark.parametrize(
    "feature_name",
    [
        pytest.param(name, id=f"Test type of {name} neurite feature")
        for name in features._NEURITE_FEATURES
    ],
)
def test_neurite_feature_types(feature_name, NEURITE):
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
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_morphology_feature_types(feature_name, MORPHOLOGY):
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
def test_population_feature_types(feature_name, POP):
    """Test population features."""
    res = features._POPULATION_FEATURES.get(feature_name)(POP)
    _check_nested_type(res)
