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

import click
import matplotlib.pyplot as plt

import neurom as nm
from neurom.apps import morph_stats, morph_check
from neurom import load_neuron
from neurom.view.plotly import draw as plotly_draw
from neurom.viewer import draw as pyplot_draw


@click.group()
@click.option('-v', '--verbose', count=True, default=0,
              help='-v for WARNING, -vv for INFO, -vvv for DEBUG')
def cli(verbose):
    """The CLI entry point."""
    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logging.basicConfig(level=level)


@cli.command()
@click.argument('input_file')
@click.option('--plane', type=click.Choice(['3d', 'xy', 'yx', 'yz', 'zy', 'xz', 'zx']),
              default='3d')
@click.option('--backend', type=click.Choice(['plotly', 'matplotlib']),
              default='matplotlib')
@click.option('-r', '--realistic-diameters/--no-realistic-diameters', default=False,
              help='Scale diameters according to the plot axis\n'
                   'Warning: Only works with the matplotlib backend')
def view(input_file, plane, backend, realistic_diameters):
    """A simple neuron viewer."""
    if backend == 'matplotlib':
        kwargs = {
            'mode': '3d' if plane == '3d' else '2d',
            'realistic_diameters': realistic_diameters,
        }
        if plane != '3d':
            kwargs['plane'] = plane
        pyplot_draw(load_neuron(input_file), **kwargs)
    else:
        plotly_draw(load_neuron(input_file), plane=plane)

    if backend == 'matplotlib':
        plt.show()


@cli.command(short_help='Morphology statistics extractor, more details at'
                        'https://neurom.readthedocs.io/en/latest/morph_stats.html')
@click.argument('datapath', required=False)
@click.option('-C', '--config', type=click.Path(exists=True, dir_okay=False),
              default=morph_stats.EXAMPLE_CONFIG, show_default=True,
              help='Configuration File')
@click.option('-o', '--output', type=click.Path(exists=False, dir_okay=False),
              help='Path to output file, if it ends in .json, a json file is created,'
                   'otherwise a csv file is created')
@click.option('-f', '--full-config', is_flag=True, default=False,
              help='If passed then --config is ignored. Compute statistics for all neurite'
                   'types, all modes and all features')
@click.option('--as-population', is_flag=True, default=False,
              help='If enabled the directory is treated as a population')
@click.option('-I', '--ignored-exceptions', help='Exception to ignore',
              type=click.Choice(morph_stats.IGNORABLE_EXCEPTIONS.keys()))
def stats(datapath, config, output, full_config, as_population, ignored_exceptions):
    """Cli for apps/morph_stats."""
    morph_stats.main(datapath, config, output, full_config, as_population, ignored_exceptions)


@cli.command(short_help='list all available features')
def features():
    """Cli to get list of available features. For backward compatibility."""
    # TODO replace it with programmatically generated Sphinx page that contains all available
    # features, also programmatically generate EXAMPLE_CONFIG on that page.
    # pylint: disable=protected-access
    print(nm.features._get_doc())


@cli.command(short_help='Perform checks on morphologies, more details at'
                        'https://neurom.readthedocs.io/en/latest/morph_check.html')
@click.argument('datapath')
@click.option('-C', '--config', type=click.Path(exists=True, dir_okay=False),
              default=morph_check.EXAMPLE_CONFIG, show_default=True,
              help='Configuration File')
@click.option('-o', '--output', type=click.Path(exists=False, dir_okay=False),
              help='Path to output json summary file')
def check(datapath, config, output):
    """Cli for apps/morph_check."""
    morph_check.main(datapath, config, output)
