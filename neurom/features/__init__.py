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
    >>> import neurom
    >>> from neurom import features
    >>> nrn = neurom.load_neuron('path/to/neuron')
    >>> ap_seg_len = features.get('segment_lengths', nrn, neurite_type=neurom.APICAL_DENDRITE)
    >>> ax_sec_len = features.get('section_lengths', nrn, neurite_type=neurom.AXON)
"""

from enum import Enum
import numpy as np

from neurom.core import Population, Neuron, Neurite
from neurom.core.neuron import iter_neurites
from neurom.core.types import NeuriteType, tree_type_checker as is_type
from neurom.exceptions import NeuroMError

_NEURITE_FEATURES = dict()
_NEURON_FEATURES = dict()
_POPULATION_FEATURES = dict()


class NameSpace(Enum):
    """The level of morphology abstraction that feature applies to."""
    NEURITE = 'neurite'
    NEURON = 'neuron'
    POPULATION = 'population'


def _get_feature_value_and_func(feature_name, obj, agg=None, **kwargs):
    """Obtain a feature from a set of morphology objects.

    Arguments:
        feature_name(string): feature to extract
        obj (Neurite|Neuron|Population): neurite, neuron or population
        agg (string|None):
        kwargs: parameters to forward to underlying worker functions

    Returns:
        A tuple (feature, func) of the feature value and its function
    """
    is_obj_list = isinstance(obj, (list, tuple))
    if not isinstance(obj, (Neurite, Neuron, Population)) and not is_obj_list:
        raise NeuroMError('Only Neurite, Neuron, Population or list, tuple of Neurite, Neuron can'
                          ' be used for feature calculation')
    if agg is not None and not hasattr(np, agg):
        raise NeuroMError('`agg` argument must an aggregating function of numpy package.')

    neurite_filter = is_type(kwargs.get('neurite_type', NeuriteType.all))
    res, feature_ = None, None

    if isinstance(obj, Neurite) or (is_obj_list and isinstance(obj[0], Neurite)):
        # input is a neurite or a list of neurites
        if feature_name in _NEURITE_FEATURES:
            feature_ = _NEURITE_FEATURES[feature_name]
            if isinstance(obj, Neurite):
                res = feature_(obj, **kwargs)
            else:
                res = [feature_(s, **kwargs) for s in obj]
    elif isinstance(obj, Neuron):
        # input is a neuron
        if feature_name in _NEURON_FEATURES:
            feature_ = _NEURON_FEATURES[feature_name]
            res = feature_(obj, **kwargs)
        elif feature_name in _NEURITE_FEATURES:
            feature_ = _NEURITE_FEATURES[feature_name]
            res = sum(feature_(s, **kwargs) for s in iter_neurites(obj, filt=neurite_filter))
    elif isinstance(obj, Population) or (is_obj_list and isinstance(obj[0], Neuron)):
        # input is a neuron population or a list of neurons
        if feature_name in _POPULATION_FEATURES:
            feature_ = _POPULATION_FEATURES[feature_name]
            res = feature_(obj, **kwargs)
        elif feature_name in _NEURON_FEATURES:
            feature_ = _NEURON_FEATURES[feature_name]
            res = [feature_(n, **kwargs) for n in obj]
        elif feature_name in _NEURITE_FEATURES:
            feature_ = _NEURITE_FEATURES[feature_name]
            res = [sum(feature_(s, **kwargs) for s in iter_neurites(n, filt=neurite_filter))
                   for n in obj]

    if res is None or feature_ is None:
        raise NeuroMError(f'Cant apply "{feature_name}" feature. Please check that it exists, '
                          'and can be applied to your input. See the features documentation page.')
    if isinstance(res, list) and agg is not None:
        res = getattr(np, agg)(res, axis=0)

    return res, feature_


def get(feature_name, obj, **kwargs):
    """Obtain a feature from a set of morphology objects.

    Features can be either Neurite features or Neuron features. For the list of Neurite features
    see :mod:`neurom.features.neuritefunc`. For the list of Neuron features see
    :mod:`neurom.features.neuronfunc`.

    Arguments:
        feature_name(string): feature to extract
        obj: a neuron, a neuron population or a neurite tree
        kwargs: parameters to forward to underlying worker functions

    Returns:
        features as a 1D, 2D or 3D numpy array.
    """
    return _get_feature_value_and_func(feature_name, obj, **kwargs)[0]


def _register_feature(namespace: NameSpace, name, func, shape):
    """Register a feature to be applied.

    Upon registration, an attribute 'shape' containing the expected
    shape of the function return is added to 'func'.

    Arguments:
        namespace(string): a namespace, see :class:`NameSpace`
        name(string): name of the feature, used to access the feature via `neurom.features.get()`.
        func(callable): single parameter function of a neurite.
        shape(tuple): the expected shape of the feature values
    """
    setattr(func, 'shape', shape)
    setattr(func, 'namespace', namespace)

    if name in _FEATURES:
        raise NeuroMError('Attempt to hide registered feature %s' % name)
    _FEATURES[name] = func


def feature(shape, namespace: NameSpace, name=None):
    """Feature decorator to automatically register the feature in the appropriate namespace.

    Arguments:
        shape(tuple): the expected shape of the feature values
        namespace(string): a namespace, see :class:`NameSpace`
        name(string): name of the feature, used to access the feature via `neurom.features.get()`.
    """
    def inner(func):
        _register_feature(namespace, name or func.__name__, func, shape)
        return func
    return inner


# These imports are necessary in order to register the features
from neurom.features import neuritefunc, neuronfunc  # noqa, pylint: disable=wrong-import-position
