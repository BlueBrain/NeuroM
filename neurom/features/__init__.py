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
    >>> m = neurom.load_morphology('path/to/morphology')
    >>> ap_seg_len = features.get('segment_lengths', m, neurite_type=neurom.APICAL_DENDRITE)
    >>> ax_sec_len = features.get('section_lengths', m, neurite_type=neurom.AXON)
"""
import operator
import collections.abc
from enum import Enum
from functools import reduce, wraps

from neurom.core import Population, Morphology, Neurite
from neurom.core.morphology import iter_neurites
from neurom.core.types import NeuriteType, tree_type_checker as is_type
from neurom.utils import flatten
from neurom.exceptions import NeuroMError

_NEURITE_FEATURES = {}
_MORPHOLOGY_FEATURES = {}
_POPULATION_FEATURES = {}


class NameSpace(Enum):
    """The level of morphology abstraction that feature applies to."""
    NEURITE = 'neurite'
    NEURON = 'morphology'
    MORPHOLOGY = 'morphology'
    POPULATION = 'population'


def get(feature_name, obj, **kwargs):
    """Obtain a feature from a set of morphology objects.

    Features can be either Neurite, Morphology or Population features. For Neurite features see
    :mod:`neurom.features.neurite`. For Morphology features see :mod:`neurom.features.morphology`.
    For Population features see :mod:`neurom.features.population`.

    Arguments:
        feature_name(string): feature to extract
        obj: a morphology, a morphology population or a neurite tree
        kwargs: parameters to forward to underlying worker functions

    Returns:
        List|Number: feature value as a list or a single number.
    """
    if isinstance(obj, Neurite):
        return _NEURITE_FEATURES[feature_name](obj, **kwargs)

    if isinstance(obj, Morphology):
        return _MORPHOLOGY_FEATURES[feature_name](obj, **kwargs)

    if isinstance(obj, Population):
        return _POPULATION_FEATURES[feature_name](obj, **kwargs)

    if isinstance(obj, collections.abc.Sequence):

        if isinstance(obj[0], Neurite):
            return [_NEURITE_FEATURES[feature_name](neurite, **kwargs) for neurite in obj]

        if isinstance(obj[0], Morphology):
            return _POPULATION_FEATURES[feature_name](obj, **kwargs)

    raise NeuroMError(f'Cant apply "{feature_name}" feature. Please check that it exists, '
                      'and can be applied to your input. See the features documentation page.'
    )


def feature(shape, namespace: NameSpace, name=None, is_reducible=True):
    """Feature decorator to automatically register the feature in the appropriate namespace.

    Arguments:
        shape(tuple): the expected shape of the feature values
        namespace(string): a namespace, see :class:`NameSpace`
        name(string): name of the feature, used to access the feature via `neurom.features.get()`.
    """

    def inner(feature_function):
        _register_feature(
            namespace=namespace,
            name=name or feature_function.__name__,
            func=feature_function,
            shape=shape,
            is_reducible=is_reducible,
        )
        return feature_function

    return inner


def _shape_dependent_flatten(obj, shape):
    return obj if shape == () else reduce(operator.concat, obj, [])


def _register_feature(namespace, name, func, shape, is_reducible=True):

    def apply_neurite_feature_to_population(func):
        def apply_to_population(population, **kwargs):
            return _shape_dependent_flatten(
                [_get_neurites_feature_value(func, shape, morph, kwargs) for morph in population],
                shape,
            )
        return apply_to_population

    def apply_neurite_feature_to_morphology(func):
        def apply_to_morphology(morphology, **kwargs):
            return _get_neurites_feature_value(func, shape, morphology, kwargs)
        return apply_to_morphology

    def apply_morphology_feature_to_population(func):
        def apply_to_population(population, **kwargs):
            return _shape_dependent_flatten(
                [func(morphology, **kwargs) for morphology in population],
                shape,
            )
        return apply_to_population

    levels = (NameSpace.NEURITE, NameSpace.MORPHOLOGY, NameSpace.POPULATION)

    levels_map = {
        NameSpace.NEURITE: _NEURITE_FEATURES,
        NameSpace.NEURON: _MORPHOLOGY_FEATURES,
        NameSpace.POPULATION: _POPULATION_FEATURES
    }

    if name in levels_map[namespace]:
        raise NeuroMError(f'A feature is already registered under "{name}"')

    levels_map[namespace][name] = func
    upstream_levels = levels[levels.index(namespace) + 1:]

    if is_reducible:

        levels_reduce = {
            (NameSpace.POPULATION, NameSpace.MORPHOLOGY): apply_morphology_feature_to_population,
            (NameSpace.POPULATION, NameSpace.NEURITE): apply_neurite_feature_to_population,
            (NameSpace.MORPHOLOGY, NameSpace.NEURITE): apply_neurite_feature_to_morphology,
        }

        for level in upstream_levels:
            if name not in levels_map[level]:
                levels_map[level][name] = levels_reduce[(level, namespace)](func)


from copy import deepcopy

def _get_neurites_feature_value(feature_, shape, obj, kwargs):
    """Collects neurite feature values appropriately to feature's shape."""

    kwargs = deepcopy(kwargs)

    if "neurite_type" in kwargs:
        neurite_type = kwargs["neurite_type"]
        del kwargs["neurite_type"]
    else:
        neurite_type = NeuriteType.all

    return reduce(
        operator.add,
        (feature_(n, **kwargs) for n in iter_neurites(obj, filt=is_type(neurite_type))),
        0 if shape == () else []
    )



# These imports are necessary in order to register the features
# noqa, pylint: disable=wrong-import-position
from neurom.features import neurite, morphology, population


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
