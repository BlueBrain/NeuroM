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

import re
import numpy as _np

from . import neuritefunc as _nrt
from . import neuronfunc as _nrn
from ..core import NeuriteType as _ntype
from ..core import iter_neurites as _ineurites
from ..core.types import tree_type_checker as _is_type
from ..exceptions import NeuroMError

FEATURES = {
    'NEURITE': {
        'total_length': _nrt.total_length,
        'total_length_per_neurite': _nrt.total_length_per_neurite,
        'neurite_lengths': _nrt.total_length_per_neurite,
        'terminal_path_lengths_per_neurite': _nrt.terminal_path_lengths_per_neurite,
        'section_lengths': _nrt.section_lengths,
        'section_term_lengths': _nrt.section_term_lengths,
        'section_bif_lengths': _nrt.section_bif_lengths,
        'neurite_volumes': _nrt.total_volume_per_neurite,
        'neurite_volume_density': _nrt.neurite_volume_density,
        'section_volumes': _nrt.section_volumes,
        'section_areas': _nrt.section_areas,
        'section_tortuosity': _nrt.section_tortuosity,
        'section_path_distances': _nrt.section_path_lengths,
        'number_of_sections': _nrt.number_of_sections,
        'number_of_sections_per_neurite': _nrt.number_of_sections_per_neurite,
        'number_of_neurites': _nrt.number_of_neurites,
        'number_of_bifurcations': _nrt.number_of_bifurcations,
        'number_of_forking_points': _nrt.number_of_forking_points,
        'number_of_terminations': _nrt.number_of_terminations,
        'section_branch_orders': _nrt.section_branch_orders,
        'section_term_branch_orders': _nrt.section_term_branch_orders,
        'section_bif_branch_orders': _nrt.section_bif_branch_orders,
        'section_radial_distances': _nrt.section_radial_distances,
        'local_bifurcation_angles': _nrt.local_bifurcation_angles,
        'remote_bifurcation_angles': _nrt.remote_bifurcation_angles,
        'partition': _nrt.bifurcation_partitions,
        'partition_asymmetry': _nrt.partition_asymmetries,
        'number_of_segments': _nrt.number_of_segments,
        'segment_lengths': _nrt.segment_lengths,
        'segment_volumes': _nrt.segment_volumes,
        'segment_radii': _nrt.segment_radii,
        'segment_midpoints': _nrt.segment_midpoints,
        'segment_taper_rates': _nrt.segment_taper_rates,
        'segment_radial_distances': _nrt.segment_radial_distances,
        'segment_meander_angles': _nrt.segment_meander_angles,
        'principal_direction_extents': _nrt.principal_direction_extents,
        'total_area_per_neurite': _nrt.total_area_per_neurite,
    },

    'NEURON': {
        'soma_radii': _nrn.soma_radii,
        'soma_surface_areas': _nrn.soma_surface_areas,
        'trunk_origin_radii': _nrn.trunk_origin_radii,
        'trunk_origin_azimuths': _nrn.trunk_origin_azimuths,
        'trunk_origin_elevations': _nrn.trunk_origin_elevations,
        'trunk_section_lengths': _nrn.trunk_section_lengths,
        'sholl_frequency': _nrn.sholl_frequency,
    }
}


def register_neurite_feature(name, func):
    '''Register a feature to be applied to neurites

    Parameters:
        name: name of the feature, used for access via get() function.
        func: single parameter function of a neurite.
    '''
    if name in FEATURES['NEURITE']:
        raise NeuroMError('Attempt to hide registered feature %s' % name)

    def _fun(neurites, neurite_type=None):
        '''Wrap neurite function from outer scope and map into list'''
        return list(func(n) for n in _ineurites(neurites, filt=_is_type(neurite_type)))

    FEATURES['NEURON'][name] = _fun


def get(feature, obj, **kwargs):
    '''Obtain a feature from a set of morphology objects

    Parameters:
        feature(string): feature to extract
        obj: a neuron, population or neurite tree
        **kwargs: parameters to forward to underlying worker functions

    Returns:
        features as a 1D or 2D numpy array.

    '''

    feature = (FEATURES['NEURITE'][feature] if feature in FEATURES['NEURITE']
               else FEATURES['NEURON'][feature])

    return _np.array(list(feature(obj, **kwargs)))


_INDENT = ' ' * 4


def _indent(string, count):
    '''indent `string` by `count` * INDENT'''
    indent = _INDENT * count
    ret = indent + string.replace('\n', '\n' + indent)
    return ret.rstrip()


def _get_doc(pattern):
    '''Get a description of all the known available features'''
    def get_docstring(func):
        '''extract doctstring, if possible'''
        docstring = ':\n'
        if func.__doc__:
            docstring += _indent(func.__doc__, 2)
        return docstring

    BLUE = '\033[94m'
    GREEN = '\033[92m'
    ENDC = '\033[0m'
    title = {'NEURITE': '\nNeurite features (neurite, neuron, neuron population):\n',
             'NEURON': '\n\033[94mNeuron features (neuron, neuron population):\033[0m\n'}

    def filtered_doc(features):
        '''Enable filtering of the doc based on pattern'''
        if pattern:
            def filt(k):   # pylint: disable=missing-docstring
                return re.search(pattern, k)
        else:
            def filt(_):  # pylint: disable=missing-docstring
                return True
        filtered_features = ((k, v) for k, v in features.items() if filt(k))
        return '\n'.join(GREEN + _INDENT + '- ' + feature + ENDC + get_docstring(func)
                         for feature, func in sorted(filtered_features))

    return '\n'.join(BLUE + title[feature_type] + ENDC + filtered_doc(FEATURES[feature_type])
                     for feature_type in ['NEURITE', 'NEURON'])


get.__doc__ += _indent('\nFeatures:\n', 1) + _indent(_get_doc(''), 2)  # pylint: disable=no-member
