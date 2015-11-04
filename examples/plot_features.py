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
'''Plot a selection of features from a morphology population'''

from neurom import ezy
from neurom.analysis import morphtree as mt
from collections import defaultdict
import json
import numpy as np


nrns = ezy.load_neurons('../Synthesizer/build/L23MC/')
sim_params = json.load(open('../Synthesizer/data/L23MC.json'))


# Neurite types of interest
NEURITES_ = (ezy.TreeType.axon,
             ezy.TreeType.apical_dendrite,
             ezy.TreeType.basal_dendrite)

# map feature names to functors that get us arrays of that
# feature, for a given tree type
GET_FEATURE = {
    'trunk_azimuth': lambda nrn, typ: [mt.trunk_azimuth(n, nrn.soma)
                                       for n in nrn.neurites if n.type == typ],
    'trunk_elevation': lambda nrn, typ: [mt.trunk_elevation(n, nrn.soma)
                                         for n in nrn.neurites if n.type == typ]
}

# For now we use all the features in the map
FEATURES = GET_FEATURE.keys()

# data structure to store results
stuff = defaultdict(lambda: defaultdict(list))

# unpack data into arrays
for nrn in nrns:
    for t in NEURITES_:
        for feat in FEATURES:
            stuff[feat][str(t).split('.')[1]].extend(
                GET_FEATURE[feat](nrn, t)
            )

# Then access the arrays of azimuths with tr_azimuth[key]
# where the keys are string representations of the tree types.

histos = []
dists = []

for feat, d in stuff.iteritems():
    for typ, data in d.iteritems():
        dist = sim_params['components'][typ][feat]
        dists.append(dist)
        print typ, feat
        print 'Distribution:', dist

        num_bins = 100
        histo = np.histogram(data, num_bins, normed=True)
        histos.append(histo)
