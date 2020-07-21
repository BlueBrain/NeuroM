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

"""Example for comparison of the same feature of multiple cells."""
import argparse
from pathlib import Path

import pylab as pl
import neurom as nm
from neurom.io.utils import get_morph_files


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Feature Comparison Between Different Cells')

    parser.add_argument('-d',
                        '--datapath',
                        help='Data directory')

    parser.add_argument('-o',
                        '--odir',
                        default='.',
                        help='Output path')

    parser.add_argument('-f',
                        '--features',
                        nargs='+',
                        help='List features separated by spaces')

    return parser.parse_args()


def stylize(ax, name, feature):
    """Stylization modifications to the plots
    """
    ax.set_ylabel(feature)
    ax.set_title(name, fontsize='small')


def histogram(neuron, feature, ax, bins=15, normed=True, cumulative=False):
    """
    Plot a histogram of the selected feature for the population of neurons.
    Plots x-axis versus y-axis on a scatter|histogram|binned values plot.

    Parameters :

        neurons : neuron list

        feature : str
        The feature of interest.

        bins : int
        Number of bins for the histogram.

        cumulative : bool
        Sets cumulative histogram on.

        ax : axes object
            the axes in which the plot is taking place
    """

    feature_values = nm.get(feature, neuron)
    # generate histogram
    ax.hist(feature_values, bins=bins, cumulative=cumulative, normed=normed)


def plot_feature(feature, cell):
    """Plot a feature
    """
    fig = pl.figure()
    ax = fig.add_subplot(111)

    if cell is not None:
        try:
            histogram(cell, feature, ax)
        except ValueError:
            pass
        stylize(ax, cell.name, feature)
    return fig


if __name__ == '__main__':
    args = parse_args()

    for morph_file in get_morph_files(args.datapath):
        nrn = nm.load_neuron(morph_file)

        for _feature in args.features:
            f = plot_feature(_feature, nrn)
            figname = "{0}_{1}.eps".format(_feature, nrn.name)
            f.savefig(Path(args.odir, figname))
            pl.close(f)
