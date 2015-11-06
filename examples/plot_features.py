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
from neurom.view import common as view_utils
from collections import defaultdict
from collections import namedtuple
import sys
import json
import argparse
import numpy as np
import scipy.stats as _st


DISTS = {
    'normal': lambda p, bins: _st.norm.pdf(bins, p['mu'], p['sigma']),
    'uniform': lambda p, bins: _st.uniform.pdf(bins, p['min'], p['max'] - p['min']),
    'constant': lambda p, bins: None
}


def bin_centers(bin_edges):
    """Return array of bin centers given an array of bin edges"""
    return (bin_edges[1:] + bin_edges[:-1]) / 2.0


def bin_widths(bin_edges):
    """Return array of bin widths given an array of bin edges"""
    return bin_edges[1:] - bin_edges[:-1]


def histo_entries(histo):
    """Calculate the number of entries in a histogram

    This is the sum of bin height * bin width
    """
    bw = bin_widths(histo[1])
    return np.sum(histo[0] * bw)


def dist_points(d, bin_edges):
    """Return an array of values according to a distribution

    Points are calculated at the center of each bin
    """
    bc = bin_centers(bin_edges)
    return DISTS[d['type']](d, bc), bc


def calc_limits(data, dist, padding=0.25):
    """Calculate a suitable range for a histogram

    Returns:
        tuple of (min, max)
    """
    _min = min(min(data), dist.get('min', sys.float_info.max))
    _max = max(max(data), dist.get('max', sys.float_info.min))

    padding = padding * (_max - _min)
    return _min - padding, _max + padding


# Neurite types of interest
NEURITES_ = (ezy.TreeType.axon,
             ezy.TreeType.apical_dendrite,
             ezy.TreeType.basal_dendrite)

# map feature names to functors that get us arrays of that
# feature, for a given tree type
GET_NEURITE_FEATURE = {
    'trunk_azimuth': lambda nrn, typ: [mt.trunk_azimuth(n, nrn.soma)
                                       for n in nrn.neurites if n.type == typ],
    'trunk_elevation': lambda nrn, typ: [mt.trunk_elevation(n, nrn.soma)
                                         for n in nrn.neurites if n.type == typ],
    'segment_length': lambda n, typ: n.get_segment_lengths(typ),
    'section_length': lambda n, typ: n.get_section_lengths(typ),
}

# For now we use all the features in the map
FEATURES = GET_NEURITE_FEATURE.keys()


def load_neurite_features(filepath):
    '''Unpack relevant data into megadict'''
    stuff = defaultdict(lambda: defaultdict(list))
    nrns = ezy.load_neurons(filepath)
    # unpack data into arrays
    for nrn in nrns:
        for t in NEURITES_:
            for feat in FEATURES:
                stuff[feat][str(t).split('.')[1]].extend(
                    GET_NEURITE_FEATURE[feat](nrn, t)
                )
    return stuff

Plot = namedtuple('Plot', 'fig, ax')


def parse_args():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(
        description='Morphology feature plotter',
        epilog='Note: Makes plots of various features and superimposes\
        input distributions')

    parser.add_argument('datapath',
                        help='Morphology data directory path')

    parser.add_argument('--mtypeconfig',
                        help='Get mtype JSON configuration file')

    return parser.parse_args()


def main(data_dir, mtype_file): # pylint: disable=too-many-locals
    '''Run the stuff'''
    # data structure to store results
    stuff = load_neurite_features(data_dir)
    sim_params = json.load(open(mtype_file))

    # load histograms, distribution parameter sets and figures into arrays.
    # To plot figures, do
    # plots[i].fig.show()
    # To modify an axis, do
    # plots[i].ax.something()

    dists = []
    _plots = []

    for feat, d in stuff.iteritems():
        for typ, data in d.iteritems():
            dist = sim_params['components'][typ][feat]
            dists.append(dist)
            print 'Type = %s, Feature = %s, Distribution = %s' % (typ, feat, dist)
            # print 'DATA', data
            num_bins = 100
            limits = calc_limits(data, dist)
            bin_edges = np.linspace(limits[0], limits[1], num_bins + 1)
            histo = np.histogram(data, bin_edges, normed=True)
            print 'PLOT LIMITS:', limits
            # print 'DATA:', data
            # print 'BIN HEIGHT', histo[0]
            plot = Plot(*view_utils.get_figure(new_fig=True, subplot=111))
            view_utils.plot_limits(plot.fig, plot.ax, xlim=limits, no_ylim=True)
            plot.ax.bar(histo[1][:-1], histo[0], width=bin_widths(histo[1]))
            dp, bc = dist_points(dist, histo[1])
            # print 'BIN CENTERS:', bc, len(bc)
            if dp is not None:
                # print 'DIST POINTS:', dp, len(dp)
                plot.ax.plot(bc, dp, 'r*')
            plot.ax.set_title('%s (%s)' % (feat, typ))
            _plots.append(plot)

    return _plots

if __name__ == '__main__':
    args = parse_args()
    plots = main(args.datapath, args.mtypeconfig)
