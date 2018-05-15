
# Copyright (c) 2016, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
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

from mock import Mock
from nose import tools as nt

from neurom.core.types import (NEURITES, NeuriteType, axon_filter,
                               dendrite_filter, tree_type_checker)


def test_tree_type_checker():
    # check that when NeuriteType.all, we accept all trees, w/o checking type
    tree_filter = tree_type_checker(NeuriteType.all)
    nt.ok_(tree_filter('fake_tree'))

    mock_tree = Mock()
    mock_tree.type = NeuriteType.axon

    # single arg
    tree_filter = tree_type_checker(NeuriteType.axon)
    nt.ok_(tree_filter(mock_tree))

    mock_tree.type = NeuriteType.basal_dendrite
    nt.ok_(not tree_filter(mock_tree))

    # multiple args
    tree_filter = tree_type_checker(NeuriteType.axon, NeuriteType.basal_dendrite)
    nt.ok_(tree_filter(mock_tree))

    tree_filter = tree_type_checker(*NEURITES)
    nt.ok_(tree_filter('fake_tree'))


def test_type_filters():
    axon = Mock()
    axon.type = NeuriteType.axon
    nt.ok_(axon_filter(axon))
    nt.ok_(not dendrite_filter(axon))

    basal_dendrite = Mock()
    basal_dendrite.type = NeuriteType.basal_dendrite
    nt.ok_(not axon_filter(basal_dendrite))
    nt.ok_(dendrite_filter(basal_dendrite))

    apical_dendrite = Mock()
    apical_dendrite.type = NeuriteType.apical_dendrite
    nt.ok_(not axon_filter(apical_dendrite))
    nt.ok_(dendrite_filter(apical_dendrite))
