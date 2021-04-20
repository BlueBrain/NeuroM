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
from pathlib import Path
import tempfile

import numpy as np
from neurom.view.common import (plt, figure_naming, get_figure, save_plot, plot_style,
                                plot_title, plot_labels, plot_legend, update_plot_limits, plot_ticks,
                                plot_sphere, plot_cylinder)
import pytest


def test_figure_naming():
    pretitle, posttitle, prefile, postfile = figure_naming(pretitle='Test', prefile="", postfile=3)
    assert pretitle == 'Test -- '
    assert posttitle == ""
    assert prefile == ""
    assert postfile == "_3"

    pretitle, posttitle, prefile, postfile = figure_naming(pretitle='', posttitle="Test", prefile="test", postfile="")
    assert pretitle == ""
    assert posttitle == " -- Test"
    assert prefile == "test_"
    assert postfile == ""


def test_get_figure():
    fig_old = plt.figure()
    fig, ax = get_figure(new_fig=False)
    assert fig == fig_old
    assert ax.get_subplotspec().colspan.start == 0
    assert ax.get_subplotspec().rowspan.start == 0

    fig1, ax1 = get_figure(new_fig=True, subplot=224)
    assert fig1 != fig_old
    assert ax1.get_subplotspec().colspan.start == 1
    assert ax1.get_subplotspec().rowspan.start == 1

    fig2, ax2 = get_figure(new_fig=True, subplot=[1, 1, 1])
    assert ax2.get_subplotspec().colspan.start == 0
    assert ax2.get_subplotspec().rowspan.start == 0
    plt.close('all')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig2, ax2 = get_figure(new_fig=False)
    assert fig2 == plt.gcf()
    assert ax2 == plt.gca()
    plt.close('all')


def test_save_plot():
    with tempfile.TemporaryDirectory() as folder:
        fig_old = plt.figure()
        output_path = Path(folder, 'subdir')
        fig = save_plot(fig_old, output_path=output_path)
        assert Path(output_path, 'Figure.png').is_file()
        plt.close('all')


def test_plot_title_default(get_fig_2d):
    fig, ax = get_fig_2d
    plot_title(ax)
    assert ax.get_title() == 'Figure'


def test_plot_title(get_fig_2d):
    fig, ax = get_fig_2d
    plot_title(ax, title='Test')
    assert ax.get_title() == 'Test'


def test_plot_labels(get_fig_2d, get_fig_3d):
    fig, ax = get_fig_2d
    plot_labels(ax)
    assert ax.get_xlabel() == 'X'
    assert ax.get_ylabel() == 'Y'

    fig, ax = get_fig_2d
    plot_labels(ax, xlabel='T', ylabel='R')
    assert ax.get_xlabel() == 'T'
    assert ax.get_ylabel() == 'R'

    fig0, ax0 = get_fig_3d
    plot_labels(ax0)
    assert ax0.get_zlabel() == 'Z'

    fig0, ax0 = get_fig_3d
    plot_labels(ax0, zlabel='T')
    assert ax0.get_zlabel() == 'T'


def test_plot_legend(get_fig_2d):
    fig, ax = get_fig_2d
    plot_legend(ax)
    legend = ax.get_legend()
    assert legend is None

    fig, ax = get_fig_2d
    ax.plot([1, 2, 3], [1, 2, 3], label='line 1')
    plot_legend(ax, no_legend=False)
    legend = ax.get_legend()
    assert legend.get_texts()[0].get_text() == 'line 1'


def test_plot_limits(get_fig_2d, get_fig_3d):
    fig, ax = get_fig_2d
    with pytest.raises(AssertionError):
        update_plot_limits(ax, white_space=0)

    fig, ax = get_fig_2d
    ax.dataLim.update_from_data_xy(((0, -100), (100, 0)))

    update_plot_limits(ax, white_space=0)
    assert ax.get_xlim() == (0, 100)
    assert ax.get_ylim() == (-100, 0)

    fig0, ax0 = get_fig_3d
    update_plot_limits(ax0, white_space=0)
    zlim0 = ax0.get_zlim()
    assert np.allclose(ax0.get_zlim(), zlim0)


def test_plot_ticks(get_fig_2d, get_fig_3d):
    fig, ax = get_fig_2d
    plot_ticks(ax)
    assert len(ax.get_xticks())
    assert len(ax.get_yticks())

    fig, ax = get_fig_2d
    plot_ticks(ax, xticks=[], yticks=[])
    assert len(ax.get_xticks()) == 0
    assert len(ax.get_yticks()) == 0

    fig, ax = get_fig_2d
    plot_ticks(ax, xticks=np.arange(3), yticks=np.arange(4))
    assert len(ax.get_xticks()) == 3
    assert len(ax.get_yticks()) == 4

    fig0, ax0 = get_fig_3d
    plot_ticks(ax0)
    assert len(ax0.get_zticks())

    fig0, ax0 = get_fig_3d
    plot_ticks(ax0, zticks=[])
    assert len(ax0.get_zticks()) == 0

    fig0, ax0 = get_fig_3d
    plot_ticks(ax0, zticks=np.arange(3))
    assert len(ax0.get_zticks()) == 3


def test_plot_style(get_fig_2d):
    fig, ax = get_fig_2d
    ax.dataLim.update_from_data_xy(((0, -100), (100, 0)))

    plot_style(fig, ax)

    assert ax.get_title() == 'Figure'
    assert ax.get_xlabel() == 'X'
    assert ax.get_ylabel() == 'Y'

    fig, ax = get_fig_2d
    ax.dataLim.update_from_data_xy(((0, -100), (100, 0)))

    plot_style(fig, ax, no_axes=True)

    assert not ax.get_frame_on()
    assert not ax.xaxis.get_visible()
    assert not ax.yaxis.get_visible()

    fig, ax = get_fig_2d
    ax.dataLim.update_from_data_xy(((0, -100), (100, 0)))
    plot_style(fig, ax, tight=True)
    assert fig.get_tight_layout()


def test_plot_cylinder():
    fig0, ax0 = get_figure(params={'projection': '3d'})
    start, end = np.array([0, 0, 0]), np.array([1, 0, 0])
    plot_cylinder(ax0, start=start, end=end,
                  start_radius=0, end_radius=10.,
                  color='black', alpha=1.)
    assert ax0.has_data()


def test_plot_sphere():
    fig0, ax0 = get_figure(params={'projection': '3d'})
    plot_sphere(ax0, [0, 0, 0], 10., color='black', alpha=1.)
    assert ax0.has_data()
