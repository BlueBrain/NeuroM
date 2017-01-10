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

'''Test neurom._bifurcationfunc functionality'''

from nose import tools as nt
from neurom.core import Section
from neurom.fst import _bifurcationfunc as bf

s0 = Section(42)
s1 = s0.add_child(Section(42))
s2 = s0.add_child(Section(42))
s3 = s0.add_child(Section(42))
s4 = s1.add_child(Section(42))
s5 = s1.add_child(Section(42))
s6 = s4.add_child(Section(42))
s7 = s4.add_child(Section(42))

a0 = Section([0.0, 0.0, 0.0])
a1 = s0.add_child(Section(42))
a2 = s0.add_child(Section(42))
a3 = s0.add_child(Section(42))
a4 = s1.add_child(Section(42))
a5 = s1.add_child(Section(42))
a6 = s4.add_child(Section(42))
a7 = s4.add_child(Section(42))

def test_bifurcation_partition():
    nt.ok_(bf.bifurcation_partition(s1) == 3.0)
    nt.ok_(bf.bifurcation_partition(s4) == 1.0)
    try:
        bf.bifurcation_partition(s0) # test if it fails for multifurcation
        bf.bifurcation_partition(s2) # test if it fails for leaf
        nt.ok_(False)
    except:
        nt.ok_(True)

def test_partition_asymmetry():
    nt.ok_(bf.partition_asymmetry(s1) == 0.5)
    nt.ok_(bf.partition_asymmetry(s4) == 0.0)
    try:
        bf.bifurcation_asymmetry(s0) # test if it fails for multifurcation
        bf.bifurcation_asymmetry(s2) # test if it fails for leaf
        nt.ok_(False)
    except:
        nt.ok_(True)



