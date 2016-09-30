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

'''Core code for morph_stats application'''
import logging
from collections import defaultdict
import numpy as np
import neurom as nm

L = logging.getLogger(__name__)


def eval_stats(values, mode):
    '''Extract a summary statistic from an array of list of values

    Parameters:
        values: numpy array of values
        mode: summary stat to extract. One of ['min', 'max', 'median', 'mean', 'std', 'raw']

    Note: fails silently if values is empty
    '''
    if mode == 'raw':
        return values.tolist()
    if mode == 'total':
        mode = 'sum'

    try:
        return getattr(np, mode)(values)
    except ValueError:
        pass


def _stat_name(feat_name, stat_mode):
    '''Set stat name based on feature name and stat mode'''
    if feat_name[-1] == 's':
        feat_name = feat_name[:-1]
    if feat_name == 'soma_radii':
        feat_name = 'soma_radius'
    if stat_mode == 'raw':
        return feat_name

    return '%s_%s' % (stat_mode, feat_name)


def extract_stats(neurons, config):
    '''Extract stats from neurons'''

    stats = defaultdict(dict)
    for ns, modes in config['neurite'].iteritems():
        for n in config['neurite_type']:
            n = _NEURITE_MAP[n]
            for mode in modes:
                stat_name = _stat_name(ns, mode)
                stats[n.name][stat_name] = eval_stats(nm.get(ns, neurons, neurite_type=n), mode)
                L.debug('Stat: %s, Neurite: %s, Type: %s',
                        stat_name, n, type(stats[n.name][stat_name]))

    for ns, modes in config['neuron'].iteritems():
        for mode in modes:
            stat_name = _stat_name(ns, mode)
            stats[stat_name] = eval_stats(nm.get(ns, neurons), mode)

    return stats


def get_header(results):
    '''Extracts the headers, using the first value in the dict as the template'''
    ret = ['name', ]
    for k, v in results.values()[0].items():
        if isinstance(v, dict):
            for metric in v.keys():
                ret.append('%s:%s' % (k, metric))
        else:
            ret.append(k)
    return ret


def generate_flattened_dict(headers, results):
    '''extract from results the fields in the headers list'''
    for name, values in results.items():
        row = []
        for header in headers:
            if header == 'name':
                row.append(name)
            elif ':' in header:
                neurite_type, metric = header.split(':')
                row.append(values[neurite_type][metric])
            else:
                row.append(values[header])
        yield row


_NEURITE_MAP = {
    'AXON': nm.AXON,
    'BASAL_DENDRITE': nm.BASAL_DENDRITE,
    'APICAL_DENDRITE': nm.APICAL_DENDRITE,
    'ALL': nm.ANY_NEURITE
}
