'''Legacy module, replaced by neurom.features.neuronfunc'''
# pylint: disable=wildcard-import,unused-wildcard-import
from warnings import warn
from neurom.features.neuronfunc import *
warn('neurom.fst._neuronfunc is being deprecated and will be removed in NeuroM v1.5.0,'
     ' replace it by neurom.features.neuronfunc', DeprecationWarning)
