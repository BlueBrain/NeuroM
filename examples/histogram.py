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

"""Simple Histogram function for multiple morphologies."""
from pathlib import Path

from itertools import chain

import numpy as np
import neurom.features
from neurom.view import matplotlib_utils
from neurom import load_morphologies


PACKAGE_DIR = Path(__file__).resolve().parent.parent


def histogram(neurons, feature, new_fig=True, subplot=111, normed=False, **kwargs):
    """
    Plot a histogram of the selected feature for the population of morphologies.
    Plots x-axis versus y-axis on a scatter|histogram|binned values plot.

    More information about the plot and how it works.

    Parameters :

        neurons : list
            List of Neurons. Single morphologies must be encapsulated in a list.

        feature : str
            The feature of interest.

        bins : int
            Number of bins for the histogram.

        cumulative : bool
            Sets cumulative histogram on.

        subplot : bool
            Default is False, which returns a matplotlib figure object. If True,
            returns a matplotlib axis object, for use as a subplot.

    Returns :

        figure_output : list
            [fig|ax, figdata, figtext]
            The first item is either a figure object (if subplot is False) or an
            axis object. The second item is an object containing the data used to
            generate the figure. The final item is text used in report generation
            as a figure legend. This text needs to be manually entered in each
            figure file.
    """

    bins = kwargs.get("bins", 25)
    cumulative = kwargs.get("cumulative", False)

    fig, ax = matplotlib_utils.get_figure(new_fig=new_fig, subplot=subplot)

    kwargs["xlabel"] = kwargs.get("xlabel", feature)

    kwargs["ylabel"] = kwargs.get("ylabel", feature + " fraction")

    kwargs["title"] = kwargs.get("title", feature + " histogram")

    feature_values = [neurom.features.get(feature, neu) for neu in neurons]

    neu_labels = [neu.name for neu in neurons]

    ax.hist(feature_values, bins=bins, cumulative=cumulative, label=neu_labels, density=normed)

    kwargs["no_legend"] = len(neu_labels) == 1

    return matplotlib_utils.plot_style(fig=fig, ax=ax, **kwargs)


def population_feature_values(pops, feature):
    """Extracts feature values per population"""
    pops_feature_values = []

    for pop in pops:
        feature_values = [neurom.features.get(feature, neu) for neu in pop]

        # ugly hack to chain in case of list of lists
        if any([isinstance(p, (list, np.ndarray)) for p in feature_values]):
            feature_values = list(chain(*feature_values))

        pops_feature_values.append(feature_values)

    return pops_feature_values


def population_histogram(pops, feature, new_fig=True, normed=False, subplot=111, **kwargs):
    """
    Plot a histogram of the selected feature for the population of morphologies.
    Plots x-axis versus y-axis on a scatter|histogram|binned values plot.

    More information about the plot and how it works.

    Parameters :

        populations : populations list

        feature : str
        The feature of interest.

        bins : int
        Number of bins for the histogram.

        cumulative : bool
        Sets cumulative histogram on.

        subplot : bool
            Default is False, which returns a matplotlib figure object. If True,
            returns a matplotlib axis object, for use as a subplot.

    Returns :

        figure_output : list
            [fig|ax, figdata, figtext]
            The first item is either a figure object (if subplot is False) or an
            axis object. The second item is an object containing the data used to
            generate the figure. The final item is text used in report generation
            as a figure legend. This text needs to be manually entered in each
            figure file.
    """

    bins = kwargs.get("bins", 25)
    cumulative = kwargs.get("cumulative", False)

    fig, ax = matplotlib_utils.get_figure(new_fig=new_fig, subplot=subplot)

    kwargs["xlabel"] = kwargs.get("xlabel", feature)

    kwargs["ylabel"] = kwargs.get("ylabel", feature + " fraction")

    kwargs["title"] = kwargs.get("title", feature + " histogram")

    pops_feature_values = population_feature_values(pops, feature)

    pops_labels = [pop.name for pop in pops]

    ax.hist(
        pops_feature_values, bins=bins, cumulative=cumulative, label=pops_labels, density=normed
    )

    kwargs["no_legend"] = len(pops_labels) == 1

    return matplotlib_utils.plot_style(fig=fig, ax=ax, **kwargs)


def main():
    pop1 = load_morphologies(Path(PACKAGE_DIR, "tests/data/valid_set"))
    pop2 = load_morphologies(Path(PACKAGE_DIR, "tests/data/valid_set"))
    population_histogram([pop1, pop2], "section_lengths")


if __name__ == "__main__":
    main()
