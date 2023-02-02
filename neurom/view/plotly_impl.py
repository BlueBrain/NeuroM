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

"""Morphology draw functions using plotly."""

import numpy as np

try:
    import plotly.graph_objs as go
    from plotly.offline import init_notebook_mode, iplot, plot
except ImportError as e:
    raise ImportError(
        'neurom[plotly] is not installed. Please install it by doing: pip install neurom[plotly]'
    ) from e

from neurom import COLS, iter_neurites, iter_segments
from neurom.core.morphology import Morphology
from neurom.utils import flatten
from neurom.view.matplotlib_impl import TREE_COLOR


def plot_morph(morph, plane='xy', inline=False, **kwargs):
    """Draw morphology in 2D.

    Args:
        morph(Morphology|Section): morphology or section
        plane(str): a string representing the 2D plane (example: 'xy')
        inline(bool): must be set to True for interactive ipython notebook plotting
        **kwargs: additional plotly keyword arguments
    """
    return _plotly(morph, plane=plane, title='morphology-2D', inline=inline, **kwargs)


def plot_morph3d(morph, inline=False, **kwargs):
    """Draw morphology in 3D.

    Args:
        morph(Morphology|Section): morphology or section
        inline(bool): must be set to True for interactive ipython notebook plotting
        **kwargs: additional plotly keyword arguments
    """
    return _plotly(morph, plane='3d', title='morphology-3D', inline=inline, **kwargs)


def _make_trace(morph, plane):
    """Create the trace to be plotted."""
    for neurite in iter_neurites(morph):
        segments = list(iter_segments(neurite))

        segs = [(s[0][COLS.XYZ], s[1][COLS.XYZ]) for s in segments]

        coords = {
            "x": list(flatten((p1[0], p2[0], None) for p1, p2 in segs)),
            "y": list(flatten((p1[1], p2[1], None) for p1, p2 in segs)),
            "z": list(flatten((p1[2], p2[2], None) for p1, p2 in segs)),
        }

        color = TREE_COLOR.get(neurite.root_node.type, 'black')
        if plane.lower() == '3d':
            plot_fun = go.Scatter3d
        else:
            plot_fun = go.Scatter
            coords = {"x": coords[plane[0]], "y": coords[plane[1]]}
        yield plot_fun(line={"color": color, "width": 2}, mode='lines', **coords)


def _fill_soma_data(morph, data, plane):
    """Fill soma data if 3D plot and returns soma_2d in all cases."""
    if not isinstance(morph, Morphology):
        return []

    if plane != '3d':
        soma_2d = [
            # filled circle
            {
                'type': 'circle',
                'xref': 'x',
                'yref': 'y',
                'fillcolor': 'rgba(50, 171, 96, 0.7)',
                'x0': morph.soma.center[0] - morph.soma.radius,
                'y0': morph.soma.center[1] - morph.soma.radius,
                'x1': morph.soma.center[0] + morph.soma.radius,
                'y1': morph.soma.center[1] + morph.soma.radius,
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
        r = morph.soma.radius
        data.append(
            go.Surface(
                x=r * np.outer(np.cos(theta), np.sin(phi)) + morph.soma.center[0],
                y=r * np.outer(np.sin(theta), np.sin(phi)) + morph.soma.center[1],
                z=r * np.outer(np.ones(point_count), np.cos(phi)) + morph.soma.center[2],
                cauto=False,
                surfacecolor=['black'] * len(phi),
                showscale=False,
            )
        )
    return soma_2d


def get_figure(morph, plane, title):
    """Returns the plotly figure containing the morphology."""
    data = list(_make_trace(morph, plane))
    axis = {
        "gridcolor": "rgb(255, 255, 255)",
        "zerolinecolor": "rgb(255, 255, 255)",
        "showbackground": True,
        "backgroundcolor": "rgb(230, 230,230)",
    }

    soma_2d = _fill_soma_data(morph, data, plane)

    layout = {
        "autosize": True,
        "title": title,
        "scene": {  # This is used for 3D plots
            "xaxis": axis,
            "yaxis": axis,
            "zaxis": axis,
            "camera": {
                "up": {"x": 0, "y": 0, "z": 1},
                "eye": {
                    "x": -1.7428,
                    "y": 1.0707,
                    "z": 0.7100,
                },
            },
            "aspectmode": "data",
        },
        "yaxis": {"scaleanchor": "x"},  # This is used for 2D plots
        "shapes": soma_2d,
    }

    res = {"data": data, "layout": layout}
    return res


def _plotly(morph, plane, title, inline, **kwargs):
    fig = get_figure(morph, plane, title)

    plot_fun = iplot if inline else plot
    if inline:
        init_notebook_mode(connected=True)  # pragma: no cover
    plot_fun(fig, filename=title + '.html', **kwargs)
    return fig
