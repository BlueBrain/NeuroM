import os
import matplotlib
if 'DISPLAY' not in os.environ:  # noqa
    matplotlib.use('Agg')  # noqa


from neurom.view import common
common._get_plt()

from neurom.view.common import plt
import pytest


@pytest.fixture
def get_fig_2d():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        yield fig, ax
    finally:
        plt.close(fig)


@pytest.fixture
def get_fig_3d():
    fig = plt.figure()
    ax = fig.add_subplot((111), projection='3d')
    try:
        yield fig, ax
    finally:
        plt.close(fig)
