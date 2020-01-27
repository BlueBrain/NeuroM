'''Legacy module, replaced by neurom.features.bifurcationfunc'''
# pylint: disable=wildcard-import,unused-wildcard-import
from warnings import warn
from neurom.features.bifurcationfunc import *

warn('neurom.fst._bifurcationfunc is being deprecated and will be removed in NeuroM v1.5.0,'
     ' replace it by neurom.features.bifurcationfunc', DeprecationWarning)
