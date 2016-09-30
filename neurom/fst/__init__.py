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

    Load a neuron

    >>> from neurom import fst
    >>> nrn = fst.load_neuron('some/data/path/morph_file.swc')

    Obtain some morphometrics

    >>> ap_seg_len = fst.get('segment_lengths', nrn, neurite_type=fst.NeuriteType.apical_dendrite)
    >>> ax_sec_len = fst.get('section_lengths', nrn, neurite_type=fst.NeuriteType.axon)

    Load neurons from a directory. This loads all SWC or HDF5 files it finds\
    and returns a list of neurons

    >>> import numpy as np  # For mean value calculation
    >>> nrns = fst.load_neurons('some/data/directory')
    >>> for nrn in nrns:
    ...     print 'mean section length', np.mean(fst.get('section_lengths', nrn))

    Iterate over all the sections in a neuron

    >>> for s in fst.iter_sections(nrn): print s.points[0][:3]

'''

import numpy as _np
from functools import partial, update_wrapper
from itertools import chain
from ..io.utils import load_neuron, load_neurons
from ._core import FstNeuron, Neurite, Section
from . import _neuritefunc as _nrt
from ._neuritefunc import iter_sections
from ._neuritefunc import iter_segments
from . import _neuronfunc as _nrn
from . import sectionfunc as _sec
from ..utils import deprecated
from ..core.population import Population
from ..core import Tree
from ..core import NeuriteType
from ..core.types import NEURITES as NEURITE_TYPES
from ..core.types import tree_type_checker as _is_type
from ..morphmath import segment_radius as seg_rad
from ..morphmath import segment_taper_rate as seg_taper
from ..morphmath import section_length as sec_len


sec_len = _sec.section_fun(sec_len)


def _iseg(nrn, neurite_type=NeuriteType.all):
    '''Build a tree type filter from a neurite type and forward to functon

    TODO:
        This should be a decorator
    '''
    return _nrt.iter_segments(nrn, neurite_filter=_is_type(neurite_type))


load_population = deprecated(msg='Use load_neurons instead.',
                             fun_name='load_population')(load_neurons)


def _as_neurons(fun, nrns, **kwargs):
    '''Get features per neuron'''
    nrns = nrns.neurons if hasattr(nrns, 'neurons') else (nrns,)
    return list(fun(n, **kwargs) for n in nrns)


NEURITEFEATURES = {
    'total_length': partial(_as_neurons, lambda n, **kw: sum(_nrt.map_sections(sec_len, n, **kw))),
    'total_length_per_neurite': _nrt.total_length_per_neurite,
    'neurite_lengths': _nrt.total_length_per_neurite,
    'section_lengths': partial(_nrt.map_sections, sec_len),
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
    'segment_radii': lambda nrn, **kwargs: [seg_rad(s) for s in _iseg(nrn, **kwargs)],
    'segment_midpoints': _nrt.segment_midpoints,
    'segment_taper_rates': lambda nrn, **kwargs: [seg_taper(s)
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


def get(feature, obj, **kwargs):
    '''Obtain a feature from a set of morphology objects

    Parameters:
        feature (string): feature to extract.
        obj: a neuron, population or neurite tree.
        **kwargs: parameters to forward to underlying worker functions.

    Returns:
        features as a 1D or 2D numpy array.
        '''

    feature = (NEURITEFEATURES[feature] if feature in NEURITEFEATURES
               else NEURONFEATURES[feature])

    return _np.array(feature(obj, **kwargs))


_SEP = '\n\t- '
_get_doc = ('\nNeurite features (neurite, neuron, neuron population):%s%s'
            '\nNeuron features (neuron, neuron population):%s%s'
            % (_SEP, _SEP.join(sorted(NEURITEFEATURES)),
               _SEP, _SEP.join(sorted(NEURONFEATURES))))

get.__doc__ += _get_doc  # pylint: disable=no-member
