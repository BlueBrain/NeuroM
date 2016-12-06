# Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
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

''' NeuroM, lightweight and fast

Examples:

    Obtain some morphometrics

    >>> ap_seg_len = fst.get('segment_lengths', nrn, neurite_type=neurom.APICAL_DENDRITE)
    >>> ax_sec_len = fst.get('section_lengths', nrn, neurite_type=neurom.AXON)

'''

import numpy as _np
from functools import partial
from itertools import chain
from ._core import FstNeuron
from . import _neuritefunc as _nrt
from . import _neuronfunc as _nrn
from . import sectionfunc as _sec
from ..core import NeuriteType as _ntype
from ..core import iter_segments as _isegments
from ..core import iter_neurites as _ineurites
from ..core.types import tree_type_checker as _is_type
from ..morphmath import segment_radius as _seg_rad
from ..morphmath import segment_taper_rate as _seg_taper
from ..morphmath import section_length as _sec_len
from ..exceptions import NeuroMError


_sec_len = _sec.section_fun(_sec_len)


def _iseg(nrn, neurite_type=_ntype.all):
    '''Build a tree type filter from a neurite type and forward to functon

    TODO:
        This should be a decorator
    '''
    return _isegments(nrn, neurite_filter=_is_type(neurite_type))


def _as_neurons(fun, nrns, **kwargs):
    '''Get features per neuron'''
    nrns = _nrn.neuron_population(nrns)
    return list(fun(n, **kwargs) for n in nrns)


NEURITEFEATURES = {
    'total_length': partial(_as_neurons, lambda n, **kw: sum(_nrt.map_sections(_sec_len, n, **kw))),
    'total_length_per_neurite': _nrt.total_length_per_neurite,
    'neurite_lengths': _nrt.total_length_per_neurite,
    'terminal_path_lengths_per_neurite': _nrt.terminal_path_lengths_per_neurite,
    'section_lengths': partial(_nrt.map_sections, _sec_len),
    'neurite_volumes': _nrt.total_volume_per_neurite,
    'neurite_volume_density': _nrt.volume_density_per_neurite,
    'section_volumes': partial(_nrt.map_sections, _sec.section_volume),
    'section_areas': partial(_nrt.map_sections, _sec.section_area),
    'section_tortuosity': partial(_nrt.map_sections, _sec.section_tortuosity),
    'section_path_distances': _nrt.section_path_lengths,
    'number_of_sections': partial(_as_neurons, _nrt.n_sections),
    'number_of_sections_per_neurite': _nrt.n_sections_per_neurite,
    'number_of_neurites': partial(_as_neurons, _nrt.n_neurites),
    'number_of_bifurcations': partial(_as_neurons, _nrt.n_bifurcation_points),
    'number_of_forking_points': partial(_as_neurons, _nrt.n_forking_points),
    'number_of_terminations': partial(_as_neurons, _nrt.n_leaves),
    'section_branch_orders': _nrt.section_branch_orders,
    'section_radial_distances': _nrt.section_radial_distances,
    'local_bifurcation_angles': _nrt.local_bifurcation_angles,
    'remote_bifurcation_angles': _nrt.remote_bifurcation_angles,
    'partition': _nrt.bifurcation_partitions,
    'number_of_segments': partial(_as_neurons, _nrt.n_segments),
    'segment_lengths': _nrt.segment_lengths,
    'segment_radii': lambda nrn, **kwargs: [_seg_rad(s) for s in _iseg(nrn, **kwargs)],
    'segment_midpoints': _nrt.segment_midpoints,
    'segment_taper_rates': lambda nrn, **kwargs: [_seg_taper(s)
                                                  for s in _iseg(nrn, **kwargs)],
    'segment_radial_distances': _nrt.segment_radial_distances,
    'segment_meander_angles': lambda nrn, **kwargs: list(chain.from_iterable(_nrt.map_sections(
        _sec.section_meander_angles, nrn, **kwargs))),
    'principal_direction_extents': _nrt.principal_direction_extents
}

NEURONFEATURES = {
    'soma_radii': _nrn.soma_radii,
    'soma_surface_areas': _nrn.soma_surface_areas,
    'trunk_origin_radii': _nrn.trunk_origin_radii,
    'trunk_origin_azimuths': _nrn.trunk_origin_azimuths,
    'trunk_origin_elevations': _nrn.trunk_origin_elevations,
    'trunk_section_lengths': _nrn.trunk_section_lengths,
}


def register_neurite_feature(name, func):
    '''Register a feature to be applied to neurites

    Parameters:
        name: name of the feature, used for access via get() function.
        func: single parameter function of a neurite.
    '''
    if name in NEURITEFEATURES:
        raise NeuroMError('Attempt to hide registered feature %s', name)

    def _fun(neurites, neurite_type=_ntype.all):
        '''Wrap neurite function from outer scope and map into list'''
        return list(func(n) for n in _ineurites(neurites, filt=_is_type(neurite_type)))

    NEURONFEATURES[name] = _fun


def get(feature, obj, **kwargs):
    '''Obtain a feature from a set of morphology objects

    Parameters:
        feature(string): feature to extract
        obj: a neuron, population or neurite tree
        **kwargs: parameters to forward to underlying worker functions

    Returns:
        features as a 1D or 2D numpy array.

    '''

    feature = (NEURITEFEATURES[feature] if feature in NEURITEFEATURES
               else NEURONFEATURES[feature])

    return _np.array(feature(obj, **kwargs))

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
        if not isinstance(func, partial) and func.__doc__:
            docstring += _indent(func.__doc__, 2)
        return docstring

    ret = ['\nNeurite features (neurite, neuron, neuron population):']
    ret.extend(_INDENT + '- ' + feature + get_docstring(func)
               for feature, func in sorted(NEURITEFEATURES.items()))

    ret.append('\nNeuron features (neuron, neuron population):')
    ret.extend(_INDENT + '- ' + feature + get_docstring(func)
               for feature, func in sorted(NEURONFEATURES.items()))

    return '\n'.join(ret)

get.__doc__ += _indent('\nFeatures:\n', 1) + _indent(_get_doc(), 2)  # pylint: disable=no-member
