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

'''Neuron helper functions'''


from itertools import product
from neurom.core.types import NeuriteType
from neurom.point_neurite.treefunc import compare_trees


def _compare_neurites(nrn_a, nrn_b, neurite_type, comp_function=compare_trees):
    '''
    Find the identical pair of neurites of determined type if existent.

    Returns:
        False if pair does not exist or not identical. True otherwise.
    '''
    neurites1 = [neu for neu in nrn_a.neurites if neu.type == neurite_type]

    neurites2 = [neu for neu in nrn_b.neurites if neu.type == neurite_type]

    if len(neurites1) == len(neurites2):
        return True if len(neurites1) == 0 and len(neurites2) == 0 else \
               len(neurites1) - sum(1 for neu1, neu2 in
                                    product(neurites1, neurites2)
                                    if comp_function(neu1, neu2)) == 0
    else:
        return False


def neurons_eq(nrn_a, nrn_b):
    '''Compare two neurons for equality'''
    return False if not isinstance(nrn_a, type(nrn_b)) else \
           all(_compare_neurites(nrn_a, nrn_b, ttype) for ttype in
               [NeuriteType.axon, NeuriteType.basal_dendrite,
                NeuriteType.apical_dendrite, NeuriteType.undefined])
