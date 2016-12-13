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
'''Feature registrationg and retrieval'''

import numpy as np

from functools import partial
from ..exceptions import NeuroMError
from .types import tree_type_checker
from . import iter_neurites
from . import NeuriteType


FEATURES = {'NEURITEFEATURES': {},
            'NEURONFEATURES': {}, }


def register_neurite_feature(name, func):
    '''Register a feature to be applied to neurites

    Parameters:
        name: name of the feature, used for access via get() function.
        func: single parameter function of a neurite.
    '''
    def _fun(neurites, neurite_type=NeuriteType.all):
        '''Wrap neurite function from outer scope and map into list'''
        return list(func(n)
                    for n in iter_neurites(neurites, filt=tree_type_checker(neurite_type)))

    register_feature('NEURITEFEATURES', name, _fun)


def register_feature(namespace, name, func):
    '''Register a feature to be applied

    Parameters:
        namespace(string):
        name(string): name of the feature, used for access via get() function.
        func(callable): single parameter function of a neurite.
    '''
    if name in FEATURES[namespace]:
        raise NeuroMError('Attempt to hide registered feature %s' % name)

    FEATURES[namespace][name] = func


def feature(func=None, namespace=None, names=''):
    '''feature decorator to automatically register the feature in the appropriate namespace'''

    if func is None:
        return partial(feature, namespace=namespace, names=names)

    names = set(names.split()) | set([func.__name__, ])
    for name in names:
        register_feature(namespace, name, func)

    return func


def get(feature_name, obj, resolution_order=('NEURITEFEATURES', 'NEURONFEATURES', ), **kwargs):
    '''Obtain a feature from a set of morphology objects

    Parameters:
        feature_name(string): feature to extract
        obj: a neuron, population or neurite tree
        **kwargs: parameters to forward to underlying worker functions

    Returns:
        features as a 1D or 2D numpy array.
    '''

    for name in resolution_order:
        if feature_name in FEATURES[name]:
            feat = FEATURES[name][feature_name]
            break
    else:
        raise NeuroMError('Unable to find feature: %s' % feature_name)

    return np.array(list(feat(obj, **kwargs)))


_INDENT = ' ' * 4


def _indent(string, count):
    '''indent `string` by `count` * INDENT'''
    indent = _INDENT * count
    ret = indent + string.replace('\n', '\n' + indent)
    return ret.rstrip()


def get_doc():
    '''Get a description of all the known available features'''
    def get_docstring(func):
        '''extract doctstring, if possible'''
        docstring = ':\n'
        if func.__doc__:
            docstring += _indent(func.__doc__, 2)
        return docstring

    ret = ['\nNeurite features (neurite, neuron, neuron population):']
    ret.extend(_INDENT + '- ' + feature + get_docstring(func)
               for feature, func in sorted(FEATURES['NEURITEFEATURES'].items()))

    ret.append('\nNeuron features (neuron, neuron population):')
    ret.extend(_INDENT + '- ' + feature + get_docstring(func)
               for feature, func in sorted(FEATURES['NEURONFEATURES'].items()))

    return '\n'.join(ret)

get.__doc__ += _indent('\nFeatures:\n', 1) + _indent(get_doc(), 2)  # pylint: disable=no-member
