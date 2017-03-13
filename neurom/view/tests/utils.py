from contextlib import contextmanager

from neurom.view import common
common._get_plt()

from neurom.view.common import plt


@contextmanager
def get_fig_2d():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 0], [1, 2], label='test')
    yield fig, ax
    plt.close(fig)


@contextmanager
def get_fig_3d():
    fig0 = plt.figure()
    ax0 = fig0.add_subplot((111), projection='3d')
    ax0.plot([0, 0], [1, 2], [2, 1])
    yield fig0, ax0
    plt.close(fig0)
