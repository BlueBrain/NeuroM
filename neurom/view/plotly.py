"""Define the public 'draw' function to be used to draw morphology using plotly."""
from __future__ import absolute_import  # prevents name clash with local plotly module
from itertools import chain

import numpy as np

try:
    import plotly.graph_objs as go
    from plotly.offline import plot, iplot, init_notebook_mode
except ImportError as e:
    raise ImportError(
        'neurom[plotly] is not installed. Please install it by doing: pip install neurom[plotly]'
    ) from e

from neurom import COLS, iter_segments, iter_neurites
from neurom.core import Neuron
from neurom.view.view import TREE_COLOR


def draw(obj, plane='3d', inline=False, **kwargs):
    """Draw the object using the given plane.

    obj (neurom.Neuron, neurom.Section): neuron or section
    plane (str): a string representing the 2D plane (example: 'xy')
                 or '3d', '3D' for a 3D view

    inline (bool): must be set to True for interactive ipython notebook plotting
    """
    if plane.lower() == '3d':
        return _plot_neuron3d(obj, inline, **kwargs)
    return _plot_neuron(obj, plane, inline, **kwargs)


def _plot_neuron(neuron, plane, inline, **kwargs):
    return _plotly(neuron, plane=plane, title='neuron-2D', inline=inline, **kwargs)


def _plot_neuron3d(neuron, inline, **kwargs):
    """Generates a figure of the neuron, that contains a soma and a list of trees."""
    return _plotly(neuron, plane='3d', title='neuron-3D', inline=inline, **kwargs)


def _make_trace(neuron, plane):
    """Create the trace to be plotted."""
    for neurite in iter_neurites(neuron):
        segments = list(iter_segments(neurite))

        segs = [(s[0][COLS.XYZ], s[1][COLS.XYZ]) for s in segments]

        coords = dict(x=list(chain.from_iterable((p1[0], p2[0], None) for p1, p2 in segs)),
                      y=list(chain.from_iterable((p1[1], p2[1], None) for p1, p2 in segs)),
                      z=list(chain.from_iterable((p1[2], p2[2], None) for p1, p2 in segs)))

        color = TREE_COLOR.get(neurite.root_node.type, 'black')
        if plane.lower() == '3d':
            plot_fun = go.Scatter3d
        else:
            plot_fun = go.Scatter
            coords = dict(x=coords[plane[0]], y=coords[plane[1]])
        yield plot_fun(
            line=dict(color=color, width=2),
            mode='lines',
            **coords
        )


def _fill_soma_data(neuron, data, plane):
    """Fill soma data if 3D plot and returns soma_2d in all cases."""
    if not isinstance(neuron, Neuron):
        return []

    if plane != '3d':
        soma_2d = [
            # filled circle
            {
                'type': 'circle',
                'xref': 'x',
                'yref': 'y',
                'fillcolor': 'rgba(50, 171, 96, 0.7)',
                'x0': neuron.soma.center[0] - neuron.soma.radius,
                'y0': neuron.soma.center[1] - neuron.soma.radius,
                'x1': neuron.soma.center[0] + neuron.soma.radius,
                'y1': neuron.soma.center[1] + neuron.soma.radius,

                'line': {
                    'color': 'rgba(50, 171, 96, 1)',
                },
            },
        ]

    else:
        soma_2d = []
        point_count = 100  # Enough points so that the surface looks like a sphere
        theta = np.linspace(0, 2 * np.pi, point_count)
        phi = np.linspace(0, np.pi, point_count)
        r = neuron.soma.radius
        data.append(
            go.Surface(
                x=r * np.outer(np.cos(theta), np.sin(phi)) + neuron.soma.center[0],
                y=r * np.outer(np.sin(theta), np.sin(phi)) + neuron.soma.center[1],
                z=r * np.outer(np.ones(point_count), np.cos(phi)) + neuron.soma.center[2],
                cauto=False,
                surfacecolor=['black'] * len(phi),
                showscale=False,
            )
        )
    return soma_2d


def get_figure(neuron, plane, title):
    """Returns the plotly figure containing the neuron."""
    data = list(_make_trace(neuron, plane))
    axis = dict(
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)'
    )

    soma_2d = _fill_soma_data(neuron, data, plane)

    layout = dict(
        autosize=True,
        title=title,
        scene=dict(  # This is used for 3D plots
            xaxis=axis, yaxis=axis, zaxis=axis,
            camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=-1.7428, y=1.0707, z=0.7100,)),
            aspectmode='data'
        ),
        yaxis=dict(scaleanchor="x"),  # This is used for 2D plots
        shapes=soma_2d,
    )

    res = dict(data=data, layout=layout)
    return res


def _plotly(neuron, plane, title, inline, **kwargs):
    fig = get_figure(neuron, plane, title)

    plot_fun = iplot if inline else plot
    if inline:
        init_notebook_mode(connected=True)  # pragma: no cover
    plot_fun(fig, filename=title + '.html', **kwargs)
    return fig
