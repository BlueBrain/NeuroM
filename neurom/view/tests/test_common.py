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
from .utils import get_fig_2d, get_fig_3d # needs to be at top to trigger matplotlib Agg backend

from pathlib import Path
from nose import tools as nt
import shutil
import tempfile

import numpy as np


from neurom.view.common import (plt, figure_naming, get_figure, save_plot, plot_style,
                                plot_title, plot_labels, plot_legend, update_plot_limits, plot_ticks,
                                plot_sphere, plot_cylinder)


def test_figure_naming():
    pretitle, posttitle, prefile, postfile = figure_naming(pretitle='Test', prefile="", postfile=3)
    nt.eq_(pretitle, 'Test -- ')
    nt.eq_(posttitle, "")
    nt.eq_(prefile, "")
    nt.eq_(postfile, "_3")

    pretitle, posttitle, prefile, postfile = figure_naming(pretitle='', posttitle="Test", prefile="test", postfile="")
    nt.eq_(pretitle, "")
    nt.eq_(posttitle, " -- Test")
    nt.eq_(prefile, "test_")
    nt.eq_(postfile, "")


def test_get_figure():
    fig_old = plt.figure()
    fig, ax = get_figure(new_fig=False)
    nt.eq_(fig, fig_old)
    nt.eq_(ax.get_subplotspec().colspan.start, 0)
    nt.eq_(ax.get_subplotspec().rowspan.start, 0)

    fig1, ax1 = get_figure(new_fig=True, subplot=224)
    nt.ok_(fig1 != fig_old)
    nt.eq_(ax1.get_subplotspec().colspan.start, 1)
    nt.eq_(ax1.get_subplotspec().rowspan.start, 1)

    fig2, ax2 = get_figure(new_fig=True, subplot=[1, 1, 1])
    nt.eq_(ax2.get_subplotspec().colspan.start, 0)
    nt.eq_(ax2.get_subplotspec().rowspan.start, 0)
    plt.close('all')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig2, ax2 = get_figure(new_fig=False)
    nt.eq_(fig2, plt.gcf())
    nt.eq_(ax2, plt.gca())
    plt.close('all')


def test_save_plot():
    with tempfile.TemporaryDirectory() as folder:
        fig_old = plt.figure()
        output_path = Path(folder, 'subdir')
        fig = save_plot(fig_old, output_path=output_path)
        nt.ok_(Path(output_path, 'Figure.png').is_file())
        plt.close('all')



def test_plot_title():
    with get_fig_2d() as (fig, ax):
        plot_title(ax)
        nt.eq_(ax.get_title(), 'Figure')

    with get_fig_2d() as (fig, ax):
        plot_title(ax, title='Test')
        nt.eq_(ax.get_title(), 'Test')


def test_plot_labels():
    with get_fig_2d() as (fig, ax):
        plot_labels(ax)
        nt.eq_(ax.get_xlabel(), 'X')
        nt.eq_(ax.get_ylabel(), 'Y')

    with get_fig_2d() as (fig, ax):
        plot_labels(ax, xlabel='T', ylabel='R')
        nt.eq_(ax.get_xlabel(), 'T')
        nt.eq_(ax.get_ylabel(), 'R')

    with get_fig_3d() as (fig0, ax0):
        plot_labels(ax0)
        nt.eq_(ax0.get_zlabel(), 'Z')

    with get_fig_3d() as (fig0, ax0):
        plot_labels(ax0, zlabel='T')
        nt.eq_(ax0.get_zlabel(), 'T')


def test_plot_legend():
    with get_fig_2d() as (fig, ax):
        plot_legend(ax)
        legend = ax.get_legend()
        nt.ok_(legend is None)

    with get_fig_2d() as (fig, ax):
        ax.plot([1, 2, 3], [1, 2, 3], label='line 1')
        plot_legend(ax, no_legend=False)
        legend = ax.get_legend()
        nt.eq_(legend.get_texts()[0].get_text(), 'line 1')


def test_plot_limits():
    with get_fig_2d() as (fig, ax):
        nt.assert_raises(AssertionError, update_plot_limits, ax, white_space=0)

    with get_fig_2d() as (fig, ax):
        ax.dataLim.update_from_data_xy(((0, -100), (100, 0)))

        update_plot_limits(ax, white_space=0)
        nt.eq_(ax.get_xlim(), (0, 100))
        nt.eq_(ax.get_ylim(), (-100, 0))

    with get_fig_3d() as (fig0, ax0):
        update_plot_limits(ax0, white_space=0)
        zlim0 = ax0.get_zlim()
        nt.ok_(np.allclose(ax0.get_zlim(), zlim0))


def test_plot_ticks():
    with get_fig_2d() as (fig, ax):
        plot_ticks(ax)
        nt.ok_(len(ax.get_xticks()))
        nt.ok_(len(ax.get_yticks()))

    with get_fig_2d() as (fig, ax):
        plot_ticks(ax, xticks=[], yticks=[])
        nt.eq_(len(ax.get_xticks()), 0)
        nt.eq_(len(ax.get_yticks()), 0)

    with get_fig_2d() as (fig, ax):
        plot_ticks(ax, xticks=np.arange(3), yticks=np.arange(4))
        nt.eq_(len(ax.get_xticks()), 3)
        nt.eq_(len(ax.get_yticks()), 4)

    with get_fig_3d() as (fig0, ax0):
        plot_ticks(ax0)
        nt.ok_(len(ax0.get_zticks()))

    with get_fig_3d() as (fig0, ax0):
        plot_ticks(ax0, zticks=[])
        nt.eq_(len(ax0.get_zticks()), 0)

    with get_fig_3d() as (fig0, ax0):
        plot_ticks(ax0, zticks=np.arange(3))
        nt.eq_(len(ax0.get_zticks()), 3)


def test_plot_style():
    with get_fig_2d() as (fig, ax):
        ax.dataLim.update_from_data_xy(((0, -100), (100, 0)))

        plot_style(fig, ax)

        nt.eq_(ax.get_title(), 'Figure')
        nt.eq_(ax.get_xlabel(), 'X')
        nt.eq_(ax.get_ylabel(), 'Y')

    with get_fig_2d() as (fig, ax):
        ax.dataLim.update_from_data_xy(((0, -100), (100, 0)))

        plot_style(fig, ax, no_axes=True)

        nt.ok_(not ax.get_frame_on())
        nt.ok_(not ax.xaxis.get_visible())
        nt.ok_(not ax.yaxis.get_visible())

    with get_fig_2d() as (fig, ax):
        ax.dataLim.update_from_data_xy(((0, -100), (100, 0)))
        plot_style(fig, ax, tight=True)
        nt.ok_(fig.get_tight_layout())


def test_plot_cylinder():
    fig0, ax0 = get_figure(params={'projection': '3d'})
    start, end = np.array([0, 0, 0]), np.array([1, 0, 0])
    plot_cylinder(ax0, start=start, end=end,
                  start_radius=0, end_radius=10.,
                  color='black', alpha=1.)
    nt.ok_(ax0.has_data())


def test_plot_sphere():
    fig0, ax0 = get_figure(params={'projection': '3d'})
    plot_sphere(ax0, [0, 0, 0], 10., color='black', alpha=1.)
    nt.ok_(ax0.has_data())
