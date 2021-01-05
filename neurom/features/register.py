'''Feature registration and retrieval'''

import numpy as np

from functools import partial


FEATURES = {'NEURITEFEATURES': {},
            'NEURONFEATURES': {}, }

# _partition_asymmetry_length = partial(_nrt.partition_asymmetries, variant='length')
# update_wrapper(_partition_asymmetry_length, _nrt.partition_asymmetries)


from neurom.features import Shape
from neurom.features import neuritefunc as _nrt
from neurom.features import neuronfunc as _nrn
from neurom.core import NeuriteType as _ntype
from neurom.core import iter_neurites as _ineurites
from neurom.core.types import tree_type_checker as _is_type
from neurom.exceptions import NeuroMError
from neurom.utils import deprecated

@deprecated(
    '`register_neurite_feature`',
    'Please use the decorator `neurom.features.register.feature` to register custom features')
def register_neurite_feature(name, func):
    """Register a feature to be applied to neurites.

    Arguments:
        name: name of the feature, used for access via get() function.
        func: single parameter function of a neurite.
    """
    if name in FEATURES['NEURITEFEATURES']:
        raise NeuroMError('Attempt to hide registered feature %s' % name)

    def _fun(neurites, neurite_type=_ntype.all):
        """Wrap neurite function from outer scope and map into list."""
        return list(func(n) for n in _ineurites(neurites, filt=_is_type(neurite_type)))

    register_feature('NEURITEFEATURES', name, _fun, shape=[Shape.Any])


def register_feature(namespace, name, func, shape):
    '''Register a feature to be applied

    Parameters:
        namespace(string):
        name(string): name of the feature, used for access via get() function.
        func(callable): single parameter function of a neurite.
    '''
    setattr(func, 'shape', shape)

    if name in FEATURES[namespace]:
        raise NeuroMError('Attempt to hide registered feature %s' % name)
    FEATURES[namespace][name] = func


def feature(shape, namespace=None, name=None):
    '''feature decorator to automatically register the feature in the appropriate namespace'''
    def inner(func):
        # Keep the old behavior that do not register those features
        # TODO: this will be changed in the next commit
        if not func.__name__.startswith('n_'):
            register_feature(namespace, name or func.__name__, func, shape)
        return func
    return inner


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
        raise NeuroMError(f'Unable to find feature: {feature_name}')

    return np.array(list(feat(obj, **kwargs)))


_INDENT = ' ' * 4


def _indent(string, count):
    '''indent `string` by `count` * INDENT'''
    indent = _INDENT * count
    ret = indent + string.replace('\n', '\n' + indent)
    return ret.rstrip()


def _get_doc():
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

get.__doc__ += _indent('\nFeatures:\n', 1) + _indent(_get_doc(), 2)  # pylint: disable=no-member
