'''Legacy module, replaced by neurom.features.sectionfunc'''
# pylint: disable=wildcard-import,unused-wildcard-import
from warnings import warn
from neurom.features.sectionfunc import *
warn('neurom.fst.sectionfunc is being deprecated and will be removed in NeuroM v1.5.0,'
     ' replace it by neurom.features.sectionfunc', DeprecationWarning)
