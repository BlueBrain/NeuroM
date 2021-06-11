import sys
from pathlib import Path

import neurom
from neurom import load_neuron
from neurom.view import plotly_impl

import mock
from numpy.testing import assert_allclose

SWC_PATH = Path(__file__).parent.parent / 'data/swc'
MORPH_FILENAME = SWC_PATH / 'Neuron.swc'
nrn = load_neuron(MORPH_FILENAME)


def _reload_module(module):
    """Force module reload."""
    import importlib
    importlib.reload(module)


def test_plotly_extra_not_installed():
    with mock.patch.dict(sys.modules, {'plotly': None}):
        try:
            _reload_module(neurom.view.plotly_impl)
            assert False, "ImportError not triggered"
        except ImportError as e:
            assert (str(e) ==
                            'neurom[plotly] is not installed. '
                            'Please install it by doing: pip install neurom[plotly]')


def test_plotly_draw_neuron3d():
    plotly_impl.plot_neuron3d(nrn, auto_open=False)
    plotly_impl.plot_neuron3d(nrn.neurites[0], auto_open=False)

    fig = plotly_impl.plot_neuron3d(load_neuron(SWC_PATH / 'simple-different-soma.swc'),
                                  auto_open=False)
    x, y, z = [fig['data'][2][key] for key in str('xyz')]
    assert_allclose(x[0, 0], 2)
    assert_allclose(x[33, 33], -1.8971143170299758)
    assert_allclose(y[0, 0], 3)
    assert_allclose(y[33, 33], 9.75)
    assert_allclose(z[0, 0], 13)
    assert_allclose(z[33, 33], 8.5)


def test_plotly_draw_neuron2d():
    plotly_impl.plot_neuron(nrn, auto_open=False)
    plotly_impl.plot_neuron(nrn.neurites[0], auto_open=False)
