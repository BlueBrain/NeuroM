"""The morph-tool command line launcher."""
import logging

import click
import matplotlib.pyplot as plt

from neurom import load_neuron
from neurom.view.plotly import draw as plotly_draw
from neurom.viewer import draw as pyplot_draw

logging.basicConfig()
logger = logging.getLogger('morph_tool')
logger.setLevel(logging.INFO)


@click.group()
def cli():
    """The CLI entry point."""


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
