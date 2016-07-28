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

'''Module for the comparison of the morphometrics between two sets of trees.'''

import numpy as np
from neurom.core import iter_neurites

from neurom.point_neurite import triplets as tri
from neurom.point_neurite import segments as seg
from neurom.point_neurite import bifurcations as bif
from neurom.point_neurite import sections as sec

feature_map = {'Meander_angle': tri.meander_angle,
               'Segment_length': seg.length,
               'Bif_angles': bif.local_angle,
               'Section_path_length': sec.end_point_path_length,
               'Section_area': sec.area,
               'Section_length': sec.length}

features = ('Meander_angle',
            'Segment_length',
            'Bif_angles',
            'Section_path_length',
            'Section_area',
            'Section_length',)


def get_features(object1, object2, flist=features):
    '''Computes features from module mod'''
    collect_all = []

    for feat in flist:

        feature_pop = np.array([l for l in iter_neurites(object1, feature_map[feat])])
        feature_neu = np.array([l for l in iter_neurites(object2, feature_map[feat])])

        # Standardization of data: (data - mean(data))/ std(data)
        m = np.mean(feature_pop)
        st = np.std(feature_pop)

        collect_all.append([(feature_pop - m) / st, (feature_neu - m) / st])

    return collect_all


def boxplots(data_all, new_fig=True, subplot=False,
             feature_titles=features, **kwargs):
    '''Plots a list of boxplots for each feature in feature_list for object 1.
    Then presents the value of object 2 for each feature as an colored objected
    in the same boxplot.

    Parameters:
        data_all:\
            A list of pairs of flattened data for each feature.
        new_fig (Optional[bool]):\
            Default is False, which returns the default matplotlib axes 111\
            If a subplot needs to be specified, it should be provided in xxx format.
        subplot (Optional[bool]):\
            Default is False, which returns a matplotlib figure object. If True,\
            returns a matplotlib axis object, for use as a subplot.

    Returns:
        fig:\
            A figure which contains the list of boxplots.
    '''
    from neurom.view import common

    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    ax.boxplot(list(np.transpose(np.array(data_all))[0]), vert=False)

    for idata, data in enumerate(data_all):
        ax.scatter(np.median(data[1]), len(data_all) - idata, s=100, color='r', marker='s')

    ax.set_yticklabels(feature_titles)

    fig, ax = common.plot_style(fig, ax, xlabel='Normalized units (dimensionless)', ylabel='',
                                title='Summarizing features', tight=True, **kwargs)

    return fig, ax
