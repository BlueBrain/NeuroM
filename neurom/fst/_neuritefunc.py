'''Legacy module, replaced by neurom.features.neuritefunc'''
# pylint: disable=wildcard-import,unused-wildcard-import
from warnings import warn
from neurom.features.neuritefunc import *
warn('neurom.fst._neuritefunc is being deprecated and will be removed in NeuroM v1.5.0,'
     ' replace it by neurom.features.neuritefunc', DeprecationWarning)
