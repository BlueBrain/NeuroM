"""Configuration for the pytest test suite."""

import warnings
from pathlib import Path

import morphio
import neurom as nm
import pytest


def _load_morph_no_warning(filename):
    all_warnings = [
        j
        for j in [getattr(morphio.Warning, i) for i in dir(morphio.Warning)]
        if isinstance(j, morphio._morphio.Warning)
    ]
    morphio.set_ignored_warning(all_warnings, True)
    morph = nm.load_morphology(filename)
    morphio.set_ignored_warning(all_warnings, False)
    return morph


@pytest.fixture
def DATA_PATH():
    return Path(__file__).parent / "data"


@pytest.fixture
def H5_PATH(DATA_PATH):
    return DATA_PATH / "h5" / "v1"


@pytest.fixture
def ASC_PATH(DATA_PATH):
    return DATA_PATH / "neurolucida"


@pytest.fixture
def SWC_PATH(DATA_PATH):
    return DATA_PATH / "swc"


@pytest.fixture
def SIMPLE_MORPHOLOGY(SWC_PATH):
    return _load_morph_no_warning(SWC_PATH / "simple.swc")


@pytest.fixture
def SIMPLE_TRUNK_MORPHOLOGY(SWC_PATH):
    return _load_morph_no_warning(SWC_PATH / "simple_trunk.swc")


@pytest.fixture
def SWC_MORPHOLOGY(SWC_PATH):
    return _load_morph_no_warning(SWC_PATH / "Neuron.swc")


@pytest.fixture
def H5_MORPHOLOGY(H5_PATH):
    return _load_morph_no_warning(H5_PATH / "Neuron.h5")


@pytest.fixture
def SWC_MORPHOLOGY_3PT(SWC_PATH):
    return _load_morph_no_warning(SWC_PATH / 'soma' / 'three_pt_soma.swc')


@pytest.fixture
def MORPHOLOGY(SWC_PATH):
    return _load_morph_no_warning(SWC_PATH / "test_morph.swc")


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
def POP(SIMPLE_MORPHOLOGY):
    return nm.load_morphologies([SIMPLE_MORPHOLOGY, SIMPLE_MORPHOLOGY])
