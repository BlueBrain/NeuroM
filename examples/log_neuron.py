#!/usr/bin/env python
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

'''Neuron building example.

An example of how to build an object representing a neuron from an SWC file
'''
import logging
from itertools import imap
from neurom import log
from neurom.io.readers import load_data
from neurom.io.utils import make_tree
from neurom.io.utils import make_neuron
from neurom.io.utils import get_soma_ids
from neurom.io.utils import get_initial_neurite_segment_ids
from neurom.core import tree
from neurom.core import neuron
from neurom.core.dataformat import COLS
from neurom.core.point import as_point


LOG = log.LOG


def file_handler(logfile, level=logging.WARNING, fmt=None):
    '''create file handler which logs INFO messages'''
    fh = logging.FileHandler(logfile)
    fh.setLevel(level)
    if fmt is not None:
        fh.setFormatter(fmt)
    return fh

LOG.addHandler(file_handler('neurons.log',
                            level=logging.DEBUG,
                            fmt=log.FMT_LONG))


def point_iter(iterator):
    '''Transform tree iterator into a point iterator

    Args:
        iterator: tree iterator for a tree holding raw data rows.
    '''
    return imap(as_point, tree.val_iter(iterator))


if __name__ == '__main__':

    filename = 'test_data/swc/Neuron.swc'

    rd = load_data(filename)

    init_seg_ids = get_initial_neurite_segment_ids(rd)

    trees = [make_tree(rd, sg) for sg in init_seg_ids]

    soma = neuron.make_soma([rd.get_row(si) for si in get_soma_ids(rd)])

    for tr in trees:
        for p in point_iter(tree.ipreorder(tr)):
            LOG.debug(p)

    LOG.info('Initial segment IDs: %s', init_seg_ids)

    nrn = neuron.Neuron(soma, trees)

    LOG.info('Neuron soma raw data % s', [r for r in nrn.soma.iter()])
    LOG.info('Neuron soma points %s', [as_point(p)
                                       for p in nrn.soma.iter()])

    LOG.info('Neuron tree init points, types')
    for tt in nrn.neurites:
        LOG.info('%s, %s', tt.value[COLS.ID], tt.value[COLS.TYPE])

    LOG.info('Making neuron 2')
    nrn2 = make_neuron(rd)
    LOG.debug('Neuron 2 soma points %s', [r for r in nrn2.soma.iter()])
    LOG.debug('Neuron 2 soma points %s', [as_point(p)
                                          for p in nrn2.soma.iter()])
    LOG.info('Neuron 2 tree init points, types')
    for tt in nrn2.neurites:
        LOG.info('%s %s', tt.value[COLS.ID], tt.value[COLS.TYPE])

    LOG.debug('Print neuron leaves as points')
    for tt in nrn2.neurites:
        for p in point_iter(tree.ileaf(tt)):
            LOG.debug(p)
