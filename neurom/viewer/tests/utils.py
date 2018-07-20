import os
import matplotlib
if 'DISPLAY' not in os.environ:  # noqa
    matplotlib.use('Agg')  # noqa


from contextlib import contextmanager

from neurom.viewer import common
common._get_plt()

from neurom.viewer.common import plt


@contextmanager
def get_fig_2d():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        yield fig, ax
    finally:
        plt.close(fig)


@contextmanager
def get_fig_3d():
    fig = plt.figure()
    ax = fig.add_subplot((111), projection='3d')
    try:
        yield fig, ax
    finally:
        plt.close(fig)
