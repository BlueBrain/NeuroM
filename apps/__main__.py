'''The morph-tool command line launcher'''
import logging

import click
from neurom import load_neuron

logging.basicConfig()
logger = logging.getLogger('morph_tool')
logger.setLevel(logging.INFO)


@click.group()
def cli():
    '''The CLI entry point'''

try:
    import plotly
    DEFAULT_BACKEND = 'plotly'
except ModuleNotFoundError:
    DEFAULT_BACKEND = 'matplotlib'


@cli.command()
@click.argument('input_file')
@click.argument('output_image', default=None, required=False)
@click.option('--plane', type=click.Choice(['3d', 'xy', 'yx', 'yz', 'zy', 'xz', 'zx']),
              default='3d')
@click.option('--backend', type=click.Choice(['plotly', 'matplotlib']),
              default=DEFAULT_BACKEND)
def view(input_file, output_image, plane, backend):
    '''A simple neuron viewer'''
    if backend == 'matplotlib':
        from neurom.viewer import draw
        kwargs = {
            'mode': '3d' if plane == '3d' else '2d',
        }
        if plane != '3d':
            kwargs['plane'] = plane
        fig, _ = draw(load_neuron(input_file), **kwargs)
        if output_image:
            fig.savefig(output_image)
    else:
        from neurom.view.plotly import draw
        fig = draw(load_neuron(input_file), plane=plane)

        if output_image:
            from plotly.io import write_image
            write_image(fig, output_image)


    if backend == 'matplotlib':
        import matplotlib.pyplot as plt
        plt.show()
