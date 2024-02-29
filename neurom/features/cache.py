"""Cache mechanism for features."""
from functools import lru_cache
from itertools import chain

# from neurom.features import _MORPHOLOGY_FEATURES, _NEURITE_FEATURES, _POPULATION_FEATURES

_NEURITE_FEATURES = {}
_MORPHOLOGY_FEATURES = {}
_POPULATION_FEATURES = {}
_CACHED_FUNCTIONS = {}


def cached_func(maxsize=None):
    """Decorator for functions than can use cache."""
    def inner(func):
        name = func.__name__
        func = lru_cache(maxsize=maxsize)(func)
        _CACHED_FUNCTIONS[name] = func
        return func
    return inner


def clear_feature_cache(features=None):
    """Clear the cache of feature functions.

    Arguments:
        features (list[str]): (optional) The names of the features whose cache should be cleared.

    Note: If the features argument is None, the caches of all feature functions are cleared.
    """
    for feature_name, feature_funcs in chain(
        _POPULATION_FEATURES.items(),
        _MORPHOLOGY_FEATURES.items(),
        _NEURITE_FEATURES.items(),
        _CACHED_FUNCTIONS.items()
    ):
        if features is not None and feature_name not in features:
            continue
        try:
            feature_funcs.get(True).cache_clear()
        except AttributeError:
            pass
