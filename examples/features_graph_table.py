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
from pathlib import Path

import pylab as pl
import neurom as nm
from neurom.io.utils import get_morph_files


PACKAGE_DIR = Path(__file__).resolve().parent.parent


def stylize(ax, name, feature):
    """Stylization modifications to the plots"""
    ax.set_ylabel(feature)
    ax.set_title(name, fontsize="small")


def histogram(neuron, feature, ax, bins=15, normed=True, cumulative=False):
    """
    Plot a histogram of the selected feature for the population of morphologies.
    Plots x-axis versus y-axis on a scatter|histogram|binned values plot.

    Parameters :

        morphologies : neuron list

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
    ax.hist(feature_values, bins=bins, cumulative=cumulative, density=normed)


def plot_feature(feature, cell):
    """Plot a feature"""
    fig = pl.figure()
    ax = fig.add_subplot(111)

    if cell is not None:
        try:
            histogram(cell, feature, ax)
        except ValueError:
            pass
        stylize(ax, cell.name, feature)
    return fig


def create_feature_plots(morphologies_dir, feature_list, output_dir):
    for morph_file in get_morph_files(morphologies_dir):
        m = nm.load_morphology(morph_file)

        for feature_name in feature_list:
            f = plot_feature(feature_name, m)
            figname = f"{feature_name}_{m.name}.eps"
            f.savefig(Path(output_dir, figname))
            pl.close(f)


def main():
    create_feature_plots(
        morphologies_dir=Path(PACKAGE_DIR, "tests/data/valid_set"),
        feature_list=["section_lengths"],
        output_dir=".",
    )


if __name__ == "__main__":
    main()
