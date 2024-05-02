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

from copy import deepcopy
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


_FEATURE_CATEGORIES = {
    NameSpace.NEURITE: _NEURITE_FEATURES,
    NameSpace.NEURON: _MORPHOLOGY_FEATURES,
    NameSpace.MORPHOLOGY: _MORPHOLOGY_FEATURES,
    NameSpace.POPULATION: _POPULATION_FEATURES,
}


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
    return get_feature_value_and_func(feature_name, obj, **kwargs)[0]


def get_feature_value_and_func(feature_name, obj, **kwargs):
    """Obtain a feature's values and corresponding function from a set of morphology objects.

    Features can be either Neurite, Morphology or Population features. For Neurite features see
    :mod:`neurom.features.neurite`. For Morphology features see :mod:`neurom.features.morphology`.
    For Population features see :mod:`neurom.features.population`.

    Arguments:
        feature_name(string): feature to extract
        obj: a morphology, a morphology population or a neurite tree
        kwargs: parameters to forward to underlying worker functions

    Returns:
        List|Number: feature value as a list or a single number.
        Callable: feature function used to calculate the value.
    """
    try:

        if isinstance(obj, Neurite):
            feature_function = _NEURITE_FEATURES[feature_name]
            return feature_function(obj, **kwargs), feature_function

        if isinstance(obj, Morphology):
            feature_function = _MORPHOLOGY_FEATURES[feature_name]
            return feature_function(obj, **kwargs), feature_function

        if isinstance(obj, Population):
            feature_function = _POPULATION_FEATURES[feature_name]
            return feature_function(obj, **kwargs), feature_function

        if isinstance(obj, collections.abc.Sequence):

            if isinstance(obj[0], Neurite):
                feature_function = _NEURITE_FEATURES[feature_name]
                return [feature_function(neu, **kwargs) for neu in obj], feature_function

            if isinstance(obj[0], Morphology):
                feature_function = _POPULATION_FEATURES[feature_name]
                return feature_function(obj, **kwargs), feature_function

    except Exception as e:

        raise NeuroMError(
            f"Cant apply '{feature_name}' feature on {type(obj)}. Please check that it exists, "
            "and can be applied to your input. See the features documentation page."
        ) from e

    raise NeuroMError(
        "Only Neurite, Morphology, Population or list, tuple of Neurite, Morphology can be used for"
        " feature calculation."
        f"Got {type(obj)} instead. See the features documentation page."
    )


def feature(shape, namespace: NameSpace, name=None):
    """Feature decorator to automatically register the feature in the appropriate namespace.

    Arguments:
        shape(tuple): the expected shape of the feature values
        namespace(string): a namespace, see :class:`NameSpace`
        name(string): name of the feature, used to access the feature via `neurom.features.get()`.
    """

    def inner(feature_function):
        _register_feature(namespace, name or feature_function.__name__, feature_function, shape)
        return feature_function

    return inner


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
    if name in _FEATURE_CATEGORIES[namespace]:
        raise NeuroMError(f'A feature is already registered under "{name}"')

    setattr(func, "shape", shape)

    _FEATURE_CATEGORIES[namespace][name] = func


def _flatten_feature(feature_value, feature_shape):
    """Flattens feature values. Applies for population features for backward compatibility."""
    return feature_value if feature_shape == () else reduce(operator.concat, feature_value, [])


def _get_neurites_feature_value(feature_, obj, kwargs):
    """Collects neurite feature values appropriately to feature's shape."""
    kwargs = deepcopy(kwargs)

    # there is no 'neurite_type' arg in _NEURITE_FEATURES
    if "neurite_type" in kwargs:
        neurite_type = kwargs["neurite_type"]
        del kwargs["neurite_type"]
    else:
        neurite_type = NeuriteType.all

    per_neurite_values = (
        feature_(n, **kwargs) for n in iter_neurites(obj, filt=is_type(neurite_type))
    )

    return reduce(operator.add, per_neurite_values, 0 if feature_.shape == () else [])


def _transform_downstream_features_to_upstream_feature_categories(features):
    """Adds each feature to all upstream feature categories, adapted for the respective objects.

    If a feature is already defined in the module of an upstream category, it is not overwritten.
    This allows to achieve both reducible features, which can be defined for instance at the neurite
    category and then automatically added to the morphology and population categories transformed to
    work with morphology and population objects respetively.

    However, if a feature is not reducible, which means that an upstream category is not comprised
    by the accumulation/sum of its components, the feature should be defined on each category
    module so that the module logic is used instead.

    After the end of this function the _NEURITE_FEATURES, _MORPHOLOGY_FEATURES, _POPULATION_FEATURES
    are updated so that all features in neurite features are also available in morphology and
    population dictionaries, and all morphology features are available in the population dictionary.

    Args:
        features: Dictionary with feature categories.

    Notes:
        Category     Upstream Categories
        --------     -----------------
        morphology   population
        neurite      morphology, population
    """
    def apply_neurite_feature_to_population(func):
        """Transforms a feature in _NEURITE_FEATURES so that it can be applied to a population.

        Args:
            func: Feature function.

        Returns:
            Transformed neurite function to be applied on a population of morphologies.
        """
        def apply_to_population(pop, **kwargs):

            per_morphology_values = [
                _get_neurites_feature_value(func, morph, kwargs) for morph in pop
            ]
            return _flatten_feature(per_morphology_values, func.shape)

        return apply_to_population

    def apply_neurite_feature_to_morphology(func):
        """Transforms a feature in _NEURITE_FEATURES so that it can be applied on neurites.

        Args:
            func: Feature function.

        Returns:
            Transformed neurite function to be applied on a morphology.
        """
        def apply_to_morphology(morph, **kwargs):
            return _get_neurites_feature_value(func, morph, kwargs)
        return apply_to_morphology

    def apply_morphology_feature_to_population(func):
        """Transforms a feature in _MORPHOLOGY_FEATURES so that it can be applied to a population.

        Args:
            func: Feature function.

        Returns:
            Transformed morphology function to be applied on a population of morphologies.
        """
        def apply_to_population(pop, **kwargs):
            per_morphology_values = [func(morph, **kwargs) for morph in pop]
            return _flatten_feature(per_morphology_values, func.shape)
        return apply_to_population

    transformations = {
        (NameSpace.POPULATION, NameSpace.MORPHOLOGY): apply_morphology_feature_to_population,
        (NameSpace.POPULATION, NameSpace.NEURITE): apply_neurite_feature_to_population,
        (NameSpace.MORPHOLOGY, NameSpace.NEURITE): apply_neurite_feature_to_morphology,
    }

    for (upstream_category, category), transformation in transformations.items():

        features = _FEATURE_CATEGORIES[category]
        upstream_features = _FEATURE_CATEGORIES[upstream_category]

        for feature_name, feature_function in features.items():

            if feature_name in upstream_features:
                continue

            upstream_features[feature_name] = transformation(feature_function)
            setattr(upstream_features[feature_name], "shape", feature_function.shape)


# These imports are necessary in order to register the features
from neurom.features import neurite, morphology, population  # noqa, pylint: disable=wrong-import-position


# Update the feature dictionaries so that features from lower categories are transformed and usable
# by upstream categories. For example, a neurite feature will be added to morphology and population
# feature dictionaries, transformed so that it works with the respective objects.
_transform_downstream_features_to_upstream_feature_categories(_FEATURE_CATEGORIES)


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
