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

import os

from neurom import load_neuron
from neurom.check import structural_checks as chk
from neurom.exceptions import MissingParentError, SomaError
from nose import tools as nt


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
SWC_PATH = os.path.join(DATA_PATH, 'swc')
H5V1_PATH = os.path.join(DATA_PATH, 'h5/v1')

# TODO:
# The origin NeuronM neuron loader was more flexible than the one in MorphIO
# As a result it was possible to load broken morphologies and call check function on them
# to see if they are broken or nothing
# The current implementation of MorphIO does not support this but it would be a good nice-to-have
# in the future

class TestIOCheckFST(object):
    def setup(self):
        self.load_neuron = load_neuron

    def test_everything_is_fine(self):

        files = [os.path.join(SWC_PATH, f)
                 for f in ['Neuron.swc',
                           'Single_apical_no_soma.swc',
                           'Single_apical.swc',
                           'Single_basal.swc',
                           'Single_axon.swc',
                           'Neuron_zero_radius.swc',
                           'sequential_trunk_off_0_16pt.swc',
                           'sequential_trunk_off_1_16pt.swc',
                           'sequential_trunk_off_42_16pt.swc',
                           'Neuron_no_missing_ids_no_zero_segs.swc']
                 ]

        for f in files:
            nt.ok_(self.load_neuron(f))



    @nt.raises(MissingParentError)
    def test_has_sequential_ids_bad_data(self):
        nt.ok_(self.load_neuron(os.path.join(SWC_PATH, 'Neuron_missing_ids.swc')))

    def test_is_single_tree_bad_data(self):
        nt.ok_(self.load_neuron(os.path.join(SWC_PATH, 'Neuron_disconnected_components.swc')))

    @nt.raises(SomaError)
    def test_multiple_somata(self):
        nt.ok_(self.load_neuron(os.path.join(SWC_PATH, 'multiple_somata.swc')))

    @nt.raises(MissingParentError)
    def test_has_no_missing_parents_bad_data(self):
        nt.ok_(self.load_neuron(os.path.join(SWC_PATH, 'Neuron_missing_parents.swc')))

    def test_has_soma_points_bad_data(self):
        f = os.path.join(SWC_PATH, 'Single_apical_no_soma.swc')
        nt.ok_(not chk.has_soma_points(self.load_neuron(f)))


    # @nt.raises(MissingParentError)
    def test_has_valid_soma_bad_data(self):
        dw = self.load_neuron(os.path.join(SWC_PATH, 'Single_apical_no_soma.swc'))
        # nt.ok_(not chk.has_valid_soma(dw))

    @nt.raises(NotImplementedError)
    def test_has_finite_radius_neurites_good_data(self):
        files = [os.path.join(SWC_PATH, f)
                 for f in ['Neuron.swc',
                           'Single_apical.swc',
                           'Single_basal.swc',
                           'Single_axon.swc']]

        files.append(os.path.join(H5V1_PATH, 'Neuron_2_branch.h5'))

        for f in files:
            ok = chk.has_all_finite_radius_neurites(self.load_neuron(f))
            nt.ok_(ok)
            nt.ok_(len(ok.info) == 0)

    @nt.raises(NotImplementedError)
    def test_has_finite_radius_neurites_bad_data(self):
        f = os.path.join(SWC_PATH, 'Neuron_zero_radius.swc')
        ok = chk.has_all_finite_radius_neurites(self.load_neuron(f))
        nt.ok_(not ok)
        nt.ok_(list(ok.info) == [194, 210, 246, 304, 493])
