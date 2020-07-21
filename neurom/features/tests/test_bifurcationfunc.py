# Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
# All rights reserved.
#
# This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the names of
#        its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Test neurom._bifurcationfunc functionality."""

from pathlib import Path
import warnings

import numpy as np
from nose import tools as nt
from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_raises

import neurom as nm
from neurom import load_neuron
from neurom.exceptions import NeuroMError

# NOTE: The 'bf' alias is used in the fst/tests modules
# Do NOT change it.
# TODO: If other neurom.features are imported,
# the should use the aliasing used in fst/tests module files
from neurom.features import bifurcationfunc as bf

DATA_PATH = Path(__file__).parent / '../../../test_data/'
SWC_PATH = Path(DATA_PATH, 'swc')
SIMPLE = nm.load_neuron(Path(SWC_PATH, 'simple.swc'))
with warnings.catch_warnings(record=True):
    SIMPLE2 = load_neuron(Path(DATA_PATH, 'neurolucida', 'not_too_complex.asc'))
    MULTIFURCATION = load_neuron(Path(DATA_PATH, 'neurolucida', 'multifurcation.asc'))


def test_local_bifurcation_angle():
    nt.ok_(bf.local_bifurcation_angle(SIMPLE.sections[1]) == np.pi)
    nt.ok_(bf.local_bifurcation_angle(SIMPLE.sections[4]) == np.pi)
    assert_raises(NeuroMError, bf.local_bifurcation_angle, SIMPLE.sections[0])


def test_remote_bifurcation_angle():
    nt.ok_(bf.remote_bifurcation_angle(SIMPLE.sections[1]) == np.pi)
    nt.ok_(bf.remote_bifurcation_angle(SIMPLE.sections[4]) == np.pi)
    assert_raises(NeuroMError, bf.local_bifurcation_angle, SIMPLE.sections[0])


def test_bifurcation_partition():
    root = SIMPLE2.neurites[0].root_node
    assert_equal(bf.bifurcation_partition(root), 3.0)
    assert_equal(bf.bifurcation_partition(root.children[0]), 1.0)

    leaf = root.children[0].children[0]
    assert_raises(NeuroMError, bf.bifurcation_partition, leaf)

    multifurcation_section = MULTIFURCATION.neurites[0].root_node.children[0]
    assert_raises(NeuroMError, bf.bifurcation_partition, multifurcation_section)


def test_partition_asymmetry():
    root = SIMPLE2.neurites[0].root_node
    assert_equal(bf.partition_asymmetry(root), 0.5)
    assert_equal(bf.partition_asymmetry(root.children[0]), 0.0)

    leaf = root.children[0].children[0]
    assert_raises(NeuroMError, bf.partition_asymmetry, leaf)

    multifurcation_section = MULTIFURCATION.neurites[0].root_node.children[0]
    assert_raises(NeuroMError, bf.partition_asymmetry, multifurcation_section)


def test_sibling_ratio():
    root = SIMPLE2.neurites[0].root_node
    assert_equal(bf.sibling_ratio(root), 1.0)
    assert_equal(bf.sibling_ratio(root.children[0]), 1.0)

    assert_equal(bf.sibling_ratio(root, method='mean'), 1.0)
    assert_equal(bf.sibling_ratio(root.children[0], method='mean'), 1.0)

    leaf = root.children[0].children[0]
    assert_raises(NeuroMError, bf.sibling_ratio, leaf)
    assert_raises(NeuroMError, bf.sibling_ratio, leaf, method='mean')

    multifurcation_section = MULTIFURCATION.neurites[0].root_node.children[0]
    assert_raises(NeuroMError, bf.sibling_ratio, multifurcation_section)

    assert_raises(ValueError, bf.sibling_ratio, root, method='unvalid-method')


def test_diameter_power_relation():
    root = SIMPLE2.neurites[0].root_node
    assert_equal(bf.diameter_power_relation(root), 2.0)
    assert_equal(bf.diameter_power_relation(root.children[0]), 2.0)

    assert_equal(bf.diameter_power_relation(root, method='mean'), 2.0)
    assert_equal(bf.diameter_power_relation(root.children[0], method='mean'), 2.0)

    leaf = root.children[0].children[0]
    assert_raises(NeuroMError, bf.diameter_power_relation, leaf)
    assert_raises(NeuroMError, bf.diameter_power_relation, leaf, method='mean')

    multifurcation_section = MULTIFURCATION.neurites[0].root_node.children[0]
    assert_raises(NeuroMError, bf.diameter_power_relation, multifurcation_section)

    assert_raises(ValueError, bf.diameter_power_relation, root, method='unvalid-method')
