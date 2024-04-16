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

"""The morph-tool command line launcher."""
import logging
from functools import partial

import click
import matplotlib.pyplot as plt

from neurom import load_morphology
from neurom.apps import morph_check, morph_stats
from neurom.view import matplotlib_impl, matplotlib_utils


@click.group()
@click.option(
    '-v', '--verbose', count=True, default=0, help='-v for WARNING, -vv for INFO, -vvv for DEBUG'
)
def cli(verbose):
    """The CLI entry point."""
    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logging.basicConfig(level=level)


@cli.command()
@click.argument('input_file')
@click.option('--3d', 'is_3d', is_flag=True)
@click.option('--plane', type=click.Choice(['xy', 'yx', 'yz', 'zy', 'xz', 'zx']), default='xy')
@click.option('--backend', type=click.Choice(['plotly', 'matplotlib']), default='matplotlib')
@click.option(
    '-r',
    '--realistic-diameters/--no-realistic-diameters',
    default=False,
    help='Scale diameters according to the plot axis\n'
    'Warning: Only works with the matplotlib backend',
)
def view(input_file, is_3d, plane, backend, realistic_diameters):
    """CLI interface to draw morphologies."""
    # pylint: disable=import-outside-toplevel
    is_matplotlib = backend == 'matplotlib'
    if is_matplotlib:
        if is_3d:
            _, ax = matplotlib_utils.get_figure(params={'projection': '3d'})
            plot = partial(matplotlib_impl.plot_morph3d, ax=ax)
        else:
            _, ax = matplotlib_utils.get_figure()
            plot = partial(
                matplotlib_impl.plot_morph,
                ax=ax,
                plane=plane,
                realistic_diameters=realistic_diameters,
            )
    else:
        from neurom.view import plotly_impl

        if is_3d:
            plot = plotly_impl.plot_morph3d
        else:
            plot = partial(plotly_impl.plot_morph, plane=plane)

    plot(load_morphology(input_file))
    if is_matplotlib:
        if not is_3d:
            plt.axis('equal')
        plt.show()


@cli.command(
    short_help='Morphology statistics extractor, more details at'
    'https://neurom.readthedocs.io/en/latest/morph_stats.html'
)
@click.argument('datapath', required=False)
@click.option(
    '-C',
    '--config',
    type=click.Path(exists=True, dir_okay=False),
    default=morph_stats.EXAMPLE_STATS_CONFIG,
    show_default=True,
    help='Configuration File',
)
@click.option(
    '-o',
    '--output',
    type=click.Path(exists=False, dir_okay=False),
    help='Path to output file, if it ends in .json, a json file is created,'
    'otherwise a csv file is created',
)
@click.option(
    '-f',
    '--full-config',
    is_flag=True,
    default=False,
    help='If passed then --config is ignored. Compute statistics for all neurite'
    'types, all modes and all features',
)
@click.option(
    '--as-population',
    is_flag=True,
    default=False,
    help='If enabled the directory is treated as a population',
)
@click.option(
    '-I',
    '--ignored-exceptions',
    help='Exception to ignore',
    type=click.Choice(morph_stats.IGNORABLE_EXCEPTIONS.keys()),
)
@click.option(
    '--use-subtrees',
    is_flag=True,
    show_default=True,
    default=False,
    help="Enable mixed subtree processing.",
)
def stats(datapath, config, output, full_config, as_population, ignored_exceptions, use_subtrees):
    """Cli for apps/morph_stats."""
    morph_stats.main(
        datapath, config, output, full_config, as_population, ignored_exceptions, use_subtrees
    )


@cli.command(
    short_help='Perform checks on morphologies, more details at'
    'https://neurom.readthedocs.io/en/latest/morph_check.html'
)
@click.argument('datapath')
@click.option(
    '-C',
    '--config',
    type=click.Path(exists=True, dir_okay=False),
    default=morph_check.EXAMPLE_CHECK_CONFIG,
    show_default=True,
    help='Configuration File',
)
@click.option(
    '-o',
    '--output',
    type=click.Path(exists=False, dir_okay=False),
    help='Path to output json summary file',
    required=True,
)
def check(datapath, config, output):
    """Cli for apps/morph_check."""
    morph_check.main(datapath, config, output)
