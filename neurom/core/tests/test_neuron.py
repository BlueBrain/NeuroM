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

from nose import tools as nt
from copy import deepcopy
from neurom.core.neuron import BaseNeuron
from neurom.core.soma import make_soma
from neurom.core.tree import ipreorder
from neurom.point_neurite.point_tree import PointTree
from neurom.point_neurite.point_tree import val_iter
from itertools import izip
import numpy as np

SOMA_SINGLE_PTS = [[11, 22, 33, 44, 1, 1, -1]]

TREE = PointTree([0.0, 0.0, 0.0, 1.0, 1, 1, 2] )
T1 = TREE.add_child(PointTree([0.0, 1.0, 0.0, 1.0, 1, 1, 2]))
T2 = T1.add_child(PointTree([0.0, 2.0, 0.0, 1.0, 1, 1, 2]))
T3 = T2.add_child(PointTree([0.0, 4.0, 0.0, 2.0, 1, 1, 2]))
T4 = T3.add_child(PointTree([0.0, 5.0, 0.0, 2.0, 1, 1, 2]))
T5 = T4.add_child(PointTree([2.0, 5.0, 0.0, 1.0, 1, 1, 2]))
T6 = T4.add_child(PointTree([0.0, 5.0, 2.0, 1.0, 1, 1, 2]))
T7 = T5.add_child(PointTree([3.0, 5.0, 0.0, 0.75, 1, 1, 2]))
T8 = T7.add_child(PointTree([4.0, 5.0, 0.0, 0.75, 1, 1, 2]))
T9 = T6.add_child(PointTree([0.0, 5.0, 3.0, 0.75, 1, 1, 2]))
T10 = T9.add_child(PointTree([0.0, 6.0, 3.0, 0.75, 1, 1, 2]))


def test_deep_copy():

    soma = make_soma([[0, 0, 0, 1, 1, 1, -1]])
    nrn1 = BaseNeuron(soma, [TREE])
    nrn2 = deepcopy(nrn1)
    check_cloned_neuron(nrn1, nrn2)


def check_cloned_neuron(nrn1, nrn2):

    # check if two neurons are identical

    # somata
    nt.assert_true(isinstance(nrn2.soma, type(nrn1.soma)))
    nt.eq_(nrn1.soma.radius, nrn2.soma.radius)

    for v1, v2 in izip(nrn1.soma.iter(), nrn2.soma.iter()):

        nt.assert_true(np.allclose(v1, v2))

    # neurites
    for neu1, neu2 in izip(nrn1.neurites, nrn2.neurites):

        nt.assert_true(isinstance(neu2, type(neu1)))

        for v1, v2 in izip(val_iter(ipreorder(neu1)), val_iter(ipreorder(neu2))):

            nt.assert_true(np.allclose(v1, v2))

    # check if the ids are different

    # somata
    nt.assert_true( nrn1.soma is not nrn2.soma)

    # neurites
    for neu1, neu2 in izip(nrn1.neurites, nrn2.neurites):

        nt.assert_true(neu1 is not neu2)

    # check if changes are propagated between neurons

    nrn2.soma.radius = 10.

    nt.ok_(nrn1.soma.radius != nrn2.soma.radius)
    # neurites
    for neu1, neu2 in izip(nrn1.neurites, nrn2.neurites):

        for v1, v2 in izip(val_iter(ipreorder(neu1)), val_iter(ipreorder(neu2))):

            v2 = np.array([-1000., -1000., -1000., 1000., -100., -100., -100.])
            nt.assert_false(any(v1 == v2))
