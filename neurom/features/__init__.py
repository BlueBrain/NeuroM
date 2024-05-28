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
    >>> m = neurom.load_morphology("tests/data/swc/Neuron.swc")
    >>> ap_seg_len = features.get('segment_lengths', m, neurite_type=neurom.APICAL_DENDRITE)
    >>> ax_sec_len = features.get('section_lengths', m, neurite_type=neurom.AXON)
"""

import operator
from enum import Enum
from functools import partial, reduce, wraps

import numpy as np

from neurom.core import Morphology, Neurite, Population
from neurom.core.morphology import iter_neurites
from neurom.core.types import NeuriteType
from neurom.core.types import tree_type_checker as is_type
from neurom.exceptions import NeuroMError

_NEURITE_FEATURES = {}
_MORPHOLOGY_FEATURES = {}
_POPULATION_FEATURES = {}


class NameSpace(Enum):
    """The level of morphology abstraction that feature applies to."""

    NEURITE = 'neurite'
    NEURON = 'morphology'
    POPULATION = 'population'


def _flatten_feature(feature_shape, feature_value):
    """Flattens feature values. Applies for population features for backward compatibility."""
    if feature_shape == ():
        return feature_value
    return reduce(operator.concat, feature_value, [])


def _get_neurites_feature_value(feature_, obj, neurite_filter, **kwargs):
    """Collects neurite feature values appropriately to feature's shape."""
    kwargs.pop('neurite_type', None)  # there is no 'neurite_type' arg in _NEURITE_FEATURES

    return reduce(
        operator.add,
        (
            iter_neurites(
                obj,
                mapfun=partial(feature_, **kwargs),
                filt=neurite_filter,
            )
        ),
        0 if feature_.shape == () else [],
    )


def _get_feature_value_and_func(feature_name, obj, **kwargs):
    """Obtain a feature from a set of morphology objects.

    Arguments:
        feature_name(string): feature to extract
        obj (Neurite|Morphology|Population): neurite, morphology or population
        kwargs: parameters to forward to underlying worker functions

    Returns:
        Tuple(List|Number, function): A tuple (feature, func) of the feature value and its function.
          Feature value can be a list or a number.
    """
    # pylint: disable=too-many-branches
    is_obj_list = isinstance(obj, (list, tuple))
    if not isinstance(obj, (Neurite, Morphology, Population)) and not is_obj_list:
        raise NeuroMError(
            "Only Neurite, Morphology, Population or list, tuple of Neurite, Morphology"
            f"can be used for feature calculation. Got: {obj}"
        )

    neurite_filter = is_type(kwargs.get('neurite_type', NeuriteType.all))
    res, feature_ = None, None

    if isinstance(obj, Neurite) or (is_obj_list and isinstance(obj[0], Neurite)):
        # input is a neurite or a list of neurites
        if feature_name in _NEURITE_FEATURES:
            if 'neurite_type' in kwargs:
                raise NeuroMError(
                    'Can not apply "neurite_type" arg to a Neurite with a neurite feature'
                )

            feature_ = _NEURITE_FEATURES[feature_name]

            if isinstance(obj, Neurite):
                res = feature_(obj, **kwargs)
            else:
                res = [feature_(s, **kwargs) for s in obj]

    elif isinstance(obj, Morphology):
        # input is a morphology
        if 'section_type' in kwargs:
            raise NeuroMError('Can not apply "section_type" arg to a Morphology')
        if feature_name in _MORPHOLOGY_FEATURES:
            feature_ = _MORPHOLOGY_FEATURES[feature_name]

            res = feature_(obj, **kwargs)

        elif feature_name in _NEURITE_FEATURES:
            feature_ = _NEURITE_FEATURES[feature_name]
            res = _get_neurites_feature_value(feature_, obj, neurite_filter, **kwargs)

    elif isinstance(obj, Population) or (is_obj_list and isinstance(obj[0], Morphology)):
        # input is a morphology population or a list of morphs
        if 'section_type' in kwargs:
            raise NeuroMError('Can not apply "section_type" arg to a Population')
        if feature_name in _POPULATION_FEATURES:
            feature_ = _POPULATION_FEATURES[feature_name]

            res = feature_(obj, **kwargs)
        elif feature_name in _MORPHOLOGY_FEATURES:
            feature_ = _MORPHOLOGY_FEATURES[feature_name]

            res = _flatten_feature(feature_.shape, [feature_(n, **kwargs) for n in obj])
        elif feature_name in _NEURITE_FEATURES:
            feature_ = _NEURITE_FEATURES[feature_name]
            res = _flatten_feature(
                feature_.shape,
                [_get_neurites_feature_value(feature_, n, neurite_filter, **kwargs) for n in obj],
            )

    if res is None or feature_ is None:
        raise NeuroMError(
            f'Cant apply "{feature_name}" feature. Please check that it exists, '
            'and can be applied to your input. See the features documentation page.'
        )

    return res, feature_


def get(feature_name, obj, **kwargs):
    """Obtain a feature from a set of morphology objects.

    Features can be either Neurite, Morphology or Population features. For Neurite features see
    :mod:`neurom.features.neurite`. For Morphology features see :mod:`neurom.features.morphology`.
    For Population features see :mod:`neurom.features.population`.

    Arguments:
        feature_name(str): feature to extract
        obj: a morphology, a morphology population or a neurite tree
        kwargs: parameters to forward to underlying worker functions

    Returns:
        List|float: feature value as a list or a single number.
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
    _map = {
        NameSpace.NEURITE: _NEURITE_FEATURES,
        NameSpace.NEURON: _MORPHOLOGY_FEATURES,
        NameSpace.POPULATION: _POPULATION_FEATURES,
    }
    if name in _map[namespace]:
        raise NeuroMError(f'A feature is already registered under "{name}"')
    _map[namespace][name] = func


def feature(shape, namespace: NameSpace, name=None):
    """Feature decorator to automatically register the feature in the appropriate namespace.

    This decorator also ensures that the results of the features are casted to built-in types.

    Arguments:
        shape(tuple): the expected shape of the feature values
        namespace(str): a namespace, see :class:`NameSpace`
        name(str): name of the feature, used to access the feature via `neurom.features.get()`.
    """

    def inner(func):
        @wraps(func)
        def scalar_wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            try:
                return res.tolist()
            except AttributeError:
                return res

        @wraps(func)
        def matrix_wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            return np.array(res).tolist()

        if shape == ():
            decorated_func = scalar_wrapper
        else:
            decorated_func = matrix_wrapper

        _register_feature(namespace, name or func.__name__, decorated_func, shape)
        return decorated_func

    return inner


# These imports are necessary in order to register the features
# pylint: disable=wrong-import-position
from neurom.features import neurite  # noqa, isort: skio

from neurom.features import morphology  # noqa, isort: skip
from neurom.features import population  # noqa, isort: skip


def _features_catalogue():
    """Returns a string with all the available builtin features."""
    indentation = "\t"
    preamble = "\n    .. Builtin Features:\n"

    def format_category(category):
        separator = "-" * len(category)
        return f"\n{indentation}{category}\n{indentation}{separator}"

    def format_features(features):
        prefix = f"\n{indentation}* "
        return prefix + f"{prefix}".join(sorted(features))

    return preamble + "".join(
        [
            format_category(category) + format_features(features) + "\n"
            for category, features in zip(
                ("Population", "Morphology", "Neurite"),
                (_POPULATION_FEATURES, _MORPHOLOGY_FEATURES, _NEURITE_FEATURES),
            )
        ]
    )


# Update the get docstring to include all available builtin features
get.__doc__ += _features_catalogue()
