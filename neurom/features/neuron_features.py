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

''' Neuron Related Features'''
import math
from decorator import decorator as _dec
from neurom.core.neuron import Neuron


@_dec
def _make_iterable(func, *args, **kwargs):
    ''' Takes care of the neuron feature input. By using this decorator the neuron functions
    can take as an input a single neuron, list of neurons or a population.
    '''
    obj = args[0]
    neurons = [obj] if isinstance(obj, Neuron) else (obj.neurons if hasattr(obj, 'neurons')
                                                     else obj)
    args = tuple([neurons] + [a for i, a in enumerate(args) if i > 0])
    return func(*args, **kwargs)


@_make_iterable
def soma_radius(neurons):
    '''Get the radius of the soma'''
    return (nrn.soma.radius for nrn in neurons)


@_make_iterable
def soma_surface_area(neurons):
    '''Get the surface area of the soma.

    Note:
        The surface area is calculated by assuming the soma is spherical.
    '''
    return (4. * math.pi * nrn.get_soma_radius() ** 2 for nrn in neurons)
