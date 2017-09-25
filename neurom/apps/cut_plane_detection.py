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
'''Module for the detection of the cut plane'''
import logging
import operator

import numpy as np
import neurom as nm

from neurom import viewer
from neurom.core import Tree
from neurom.core.dataformat import COLS


L = logging.getLogger(__name__)


def _create_1d_distributions(points, bin_width):
    '''Create point histograms along each axis

    Parameters:
        points: a np.ndarray of points
        bin_width: the bin width

    Returns: a dict of the X, Y and Z 1D histograms'''
    min_, max_ = np.min(points, axis=0), np.max(points, axis=0)
    hist = dict()
    for i, plane in enumerate('XYZ'):
        # Two binnings with same spacing but different offsets
        # first is attached to min_[i], last is attached to max_[i]
        binning_first_slice = np.arange(min_[i], max_[i] + bin_width, bin_width)
        binning_last_slice = np.arange(max_[i], min_[i] - bin_width, -bin_width)[::-1]
        hist[plane] = (np.histogram(points[:, i], bins=binning_first_slice),
                       np.histogram(points[:, i], bins=binning_last_slice))
    return hist


def _get_probabilities(hist):
    '''Returns -log(p) where p is the a posteriori probabilities of the observed values
    in the bins X min, X max, Y min, Y max, Z min, Z max

    Parameters:
        hist: a dict of the X, Y and Z 1D histograms

    Returns: a dict of -log(p) values'''
    def minus_log_p(mu):
        '''Compute -Log(p) where p is the a posteriori probability to observe 0 counts
        in bin given than the mean value was "mu":
        The number of counts follows a Poisson law so the result is simply... mu
        demo: p(k|mu) = exp(-mu) * mu**k / k!
              p(0|mu) = exp(-mu)
              -log(p) = mu
        '''
        return mu

    for plane, (left, right) in hist.items():
        yield plane, 0, minus_log_p(left[0][0]), left
        yield plane, -1, minus_log_p(right[0][-1]), right


def _get_cut_leaves(neuron, cut_plane_and_position, tolerance):
    '''Returns leaves within cut plane tolerance'''
    cut_plane, position = cut_plane_and_position
    leaves = np.array([leaf.points[-1, COLS.XYZ]
                       for neurite in neuron.neurites
                       for leaf in nm.iter_sections(neurite, iterator_type=Tree.ileaf)])
    idx = 'XYZ'.find(cut_plane)
    return leaves[np.abs(leaves[:, idx] - position) < tolerance]


def draw_neuron(neuron, cut_plane_position, cut_leaves=None):
    '''Draw the neuron in the xy, yz and xz planes.

    Parameters:
        neuron: a Neuron
        cut_plane_position: a tuple (plane, position) like ('Z', 27)
                            that can be specified to add the cut plane on
                            the relevant plots
        cut_leaves: leaves to be highlighted by blue circles
    '''
    figures = dict()
    for draw_plane in ['yz', 'xy', 'xz']:
        fig, axes = viewer.draw(neuron, plane=draw_plane)
        figures[draw_plane] = (fig, axes)

        if cut_leaves is not None:
            import matplotlib
            slice_index = {'yz': [1, 2], 'xy': [0, 1], 'xz': [0, 2]}
            for point in cut_leaves:
                point_2d = point[slice_index[draw_plane]]
                axes.add_artist(matplotlib.patches.Circle(point_2d, radius=2))

        if cut_plane_position:
            cut_plane, position = cut_plane_position
            for axis, line_func in zip(draw_plane, ['axvline', 'axhline']):
                if axis == cut_plane.lower():
                    getattr(axes, line_func)(position)
    return figures


def draw_dist_1d(hist, cut_position=None):
    '''Draw the 1D histograms, optionally also drawing a vertical line at cut_position'''
    import matplotlib.pyplot as plt
    fig = plt.figure()

    h, bins = hist
    plt.hist(bins[:-1], bins=bins, weights=h, label='Point distribution')
    plt.xlabel('Coordinate')
    plt.ylabel('Number of points')

    if cut_position:
        plt.gca().axvline(cut_position, color='r', label='Cut plane')
        plt.legend()
    return {'distrib_1d': (fig, plt.gca())}


def _get_status(minus_log_p):
    '''Returns ok if the probability that there is a cut plane is high enough'''
    _THRESHOLD = 50
    if minus_log_p < _THRESHOLD:
        return 'The probability that there is in fact NO cut plane is high: -log(p) = {0} !'\
            .format(minus_log_p)
    return 'ok'


def find_cut_plane(neuron, bin_width=10, display=False):
    """Find the cut plane

    Parameters:
        neuron: a Neuron object
        bin_width: The size of the binning
        display: where or not to display the control plots
                 Note: It is the user responsability to call matplotlib.pyplot.show()

    Returns:
        A dictionary with the following items:
        status: 'ok' if everything went write, else an informative string
        cut_plane: a tuple (plane, position) where 'plane' is 'X', 'Y' or 'Z'
                   and 'position' is the position
        cut_leaves: an np.array of all termination points in the cut plane
        figures: if 'display' option was used, a dict where values are tuples (fig, ax)
                 for each figure
        details: A dict currently only containing -LogP of the bin where the cut plane was found

    1) The distribution of all points along X, Y and Z is computed
       and put into 3 histograms.

    2) For each histogram we look at the first and last empty bins
       (ie. the last bin before the histogram starts rising,
       and the first after it reaches zero again). Under the assumption
       that there is no cut plane, the posteriori probability
       of observing this empty bin given the value of the not-empty
       neighbour bin is then computed.
    3) The lowest probability of the 6 probabilities (2 for each axes)
       corresponds to the cut plane"""

    points = np.array([point
                       for neurite in (neuron.neurites or [])
                       for section in nm.iter_sections(neurite)
                       for point in section.points])
    if not points.size:
        return {'cut_leaves': None, 'status': "Empty neuron", 'cut_plane': None, 'details': None}

    hist = _create_1d_distributions(points, bin_width)

    cut_plane, side, minus_log_p, histo = max(_get_probabilities(hist), key=operator.itemgetter(2))

    cut_position = histo[1][side]
    cut_leaves = _get_cut_leaves(neuron, (cut_plane, cut_position), bin_width)

    result = {'cut_leaves': cut_leaves,
              'status': _get_status(minus_log_p),
              'details': {'-LogP': minus_log_p},
              'cut_plane': (cut_plane, cut_position)}

    if display:
        result['figures'] = dict()
        result['figures'].update(draw_neuron(neuron, (cut_plane, cut_position), cut_leaves))
        result['figures'].update(draw_dist_1d(histo, cut_position))
        L.info('Trigger the plot display with: matplotlib.pyplot.show()')

    return result
