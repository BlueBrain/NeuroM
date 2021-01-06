# Copyright (c) 2020, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
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

"""NeuroM, lightweight and fast.

Examples:
    Obtain some morphometrics

    >>> ap_seg_len = features.get('segment_lengths', nrn, neurite_type=neurom.APICAL_DENDRITE)
    >>> ax_sec_len = features.get('section_lengths', nrn, neurite_type=neurom.AXON)
"""

import numpy as np

from neurom.core import NeuriteType as _ntype
from neurom.core import iter_neurites as _ineurites
from neurom.core.types import tree_type_checker as _is_type
from neurom.exceptions import NeuroMError
from neurom.utils import deprecated

NEURITEFEATURES = dict()
NEURONFEATURES = dict()


@deprecated(
    '`register_neurite_feature`',
    'Please use the decorator `neurom.features.register.feature` to register custom features')
def register_neurite_feature(name, func):
    """Register a feature to be applied to neurites.

    .. deprecated:: 1.6.0

    Arguments:
        name: name of the feature, used for access via get() function.
        func: single parameter function of a neurite.

    """
    if name in NEURITEFEATURES:
        raise NeuroMError('Attempt to hide registered feature %s' % name)

    def _fun(neurites, neurite_type=_ntype.all):
        """Wrap neurite function from outer scope and map into list."""
        return list(func(n) for n in _ineurites(neurites, filt=_is_type(neurite_type)))

    _register_feature('NEURITEFEATURES', name, _fun, shape=(...))


def get(feature_name, obj, **kwargs):
    """Obtain a feature from a set of morphology objects.

    Arguments:
        feature(string): feature to extract
        obj: a neuron, population or neurite tree
        kwargs: parameters to forward to underlying worker functions

    Returns:
        features as a 1D or 2D numpy array.
    """
    for feature_list in (NEURITEFEATURES, NEURONFEATURES):
        if feature_name in feature_list:
            feat = feature_list[feature_name]
            break
    else:
        raise NeuroMError(f'Unable to find feature: {feature_name}')

    return np.array(list(feat(obj, **kwargs)))


_INDENT = ' ' * 4


def _indent(string, count):
    """Indent `string` by `count` * INDENT."""
    indent = _INDENT * count
    ret = indent + string.replace('\n', '\n' + indent)
    return ret.rstrip()


def _get_doc():
    """Get a description of all the known available features."""
    def get_docstring(func):
        """Extract doctstring, if possible."""
        docstring = ':\n'
        if func.__doc__:
            docstring += _indent(func.__doc__, 2)
        return docstring

    ret = ['\nNeurite features (neurite, neuron, neuron population):']
    ret.extend(_INDENT + '- ' + feature + get_docstring(func)
               for feature, func in sorted(NEURITEFEATURES.items()))

    ret.append('\nNeuron features (neuron, neuron population):')
    ret.extend(_INDENT + '- ' + feature + get_docstring(func)
               for feature, func in sorted(NEURONFEATURES.items()))

    return '\n'.join(ret)


get.__doc__ += _indent('\nFeatures:\n', 1) + _indent(_get_doc(), 2)  # pylint: disable=no-member


def _register_feature(namespace, name, func, shape):
    """Register a feature to be applied.

    Upon registration, an attribute 'shape' containing the expected
    shape of the function return is added to 'func'.

    Arguments:
        namespace(string): a namespace (must be 'NEURITEFEATURES' or 'NEURONFEATURES')
        name(string): name of the feature, used to access the feature via `neurom.features.get()`.
        func(callable): single parameter function of a neurite.
        shape(tuple): the expected shape of the feature values
    """
    setattr(func, 'shape', shape)

    assert namespace in {'NEURITEFEATURES', 'NEURONFEATURES'}
    feature_list = globals()[namespace]

    if name in feature_list:
        raise NeuroMError('Attempt to hide registered feature %s' % name)
    feature_list[name] = func


def feature(shape, namespace=None, name=None):
    """Feature decorator to automatically register the feature in the appropriate namespace.

    Arguments:
        shape(tuple): the expected shape of the feature values
        namespace(string): a namespace (must be 'NEURITEFEATURES' or 'NEURONFEATURES')
        name(string): name of the feature, used to access the feature via `neurom.features.get()`.
    """
    def inner(func):
        # Keep the old behavior that do not register those features
        # TODO: this will be changed in the next commit
        if not func.__name__.startswith('n_'):
            _register_feature(namespace, name or func.__name__, func, shape)
        return func
    return inner


# These imports are necessary in order to register the features
from neurom.features import neuritefunc, neuronfunc  # noqa, pylint: disable=wrong-import-position
