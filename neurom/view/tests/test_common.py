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

from nose import tools as nt
from neurom.view.common import plt
from neurom.view.common import figure_naming
from neurom.view.common import get_figure
from neurom.view.common import save_plot
from neurom.view.common import plot_style
from neurom.view.common import plot_title
from neurom.view.common import plot_labels
from neurom.view.common import plot_legend
from neurom.view.common import plot_limits
from neurom.view.common import plot_ticks
from neurom.view.common import plot_sphere
from neurom.view.common import get_color
from neurom.core.types import NeuriteType

import os
import numpy as np

fig_name = 'Figure.png'
fig_dir  = './Testing_save_in_directory/'

def test_figure_naming():
    pretitle, posttitle, prefile, postfile = figure_naming(pretitle='Test', posttitle=None, prefile="", postfile=3)
    nt.ok_(pretitle == 'Test -- ')
    nt.ok_(posttitle == "")
    nt.ok_(prefile == "")
    nt.ok_(postfile == "_3")
    pretitle, posttitle, prefile, postfile = figure_naming(pretitle='', posttitle="Test", prefile="test", postfile="")
    nt.ok_(pretitle == "")
    nt.ok_(posttitle == " -- Test")
    nt.ok_(prefile == "test_")
    nt.ok_(postfile == "")

def test_get_figure():
    fig_old = plt.figure()
    fig, ax = get_figure(new_fig=False, subplot=False)
    nt.ok_(fig == fig_old)
    nt.ok_(ax.colNum == 0)
    nt.ok_(ax.rowNum == 0)
    fig1, ax1 = get_figure(new_fig=True, subplot=224)
    nt.ok_(fig1 != fig_old)
    nt.ok_(ax1.colNum == 1)
    nt.ok_(ax1.rowNum == 1)
    fig = get_figure(new_fig=True, no_axes=True)
    nt.ok_(type(fig) == plt.Figure)
    fig2, ax2 = get_figure(new_fig=True, subplot=[1,1,1])
    nt.ok_(ax2.colNum == 0)
    nt.ok_(ax2.rowNum == 0)
    plt.close('all')
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    fig2, ax2 = get_figure(new_fig=False, new_axes=False)
    nt.ok_(fig2 == plt.gcf())
    nt.ok_(ax2 == plt.gca())
    plt.close('all')

def test_save_plot():
    if os.path.isfile(fig_name):
        os.remove(fig_name)
    fig_old = plt.figure()
    fig = save_plot(fig_old)
    nt.ok_(os.path.isfile(fig_name)==True)
    os.remove(fig_name)
    if os.path.isdir(fig_dir):
        for data in os.listdir(fig_dir):
            os.remove(fig_dir + data)
        os.rmdir(fig_dir)
    fig = save_plot(fig_old, output_path=fig_dir)
    nt.ok_(os.path.isfile(fig_dir + fig_name)==True)
    os.remove(fig_dir + fig_name)
    os.rmdir(fig_dir)
    plt.close('all')

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.plot([0,0], [1,2], label='test')
xlim = ax.get_xlim()
ylim = ax.get_ylim()

fig0 = plt.figure()
ax0  = fig0.add_subplot((111), projection='3d')
ax0.plot([0,0], [1,2], [2,1])
xlim0 = ax0.get_xlim()
ylim0 = ax0.get_ylim()
zlim0 = ax0.get_zlim()

def test_plot_title():
    fig1, ax1 = plot_title(fig, ax)
    nt.ok_(ax1.get_title() == 'Figure')
    fig1, ax1 = plot_title(fig, ax, title='Test')
    nt.ok_(ax1.get_title() == 'Test')

def test_plot_labels():
    fig1, ax1 = plot_labels(fig, ax)
    nt.ok_(ax1.get_xlabel() == 'X')
    nt.ok_(ax1.get_ylabel() == 'Y')
    fig1, ax1 = plot_labels(fig, ax, xlabel='T', ylabel='R')
    nt.ok_(ax1.get_xlabel() == 'T')
    nt.ok_(ax1.get_ylabel() == 'R')
    fig2, ax2 = plot_labels(fig0, ax0)
    nt.ok_(ax2.get_zlabel() == 'Z')
    fig2, ax2 = plot_labels(fig0, ax0, zlabel='T')
    nt.ok_(ax2.get_zlabel() == 'T')

def test_plot_legend():
    fig1, ax1 = plot_legend(fig, ax)
    legend = ax1.get_legend()
    nt.ok_(legend is None)
    fig1, ax1 = plot_legend(fig, ax, no_legend=False)
    legend = ax1.get_legend()
    nt.ok_(legend.get_texts()[0].get_text() == 'test')

def test_plot_limits():
    fig1, ax1 = plot_limits(fig, ax)
    nt.ok_(ax1.get_xlim() == xlim)
    nt.ok_(ax1.get_ylim() == ylim)
    fig1, ax1 = plot_limits(fig, ax, xlim=(0,100), ylim=(-100,0))
    nt.ok_(ax1.get_xlim() == (0,100))
    nt.ok_(ax1.get_ylim() == (-100,0))
    fig2, ax2 = plot_limits(fig0, ax0)
    nt.ok_(np.allclose(ax2.get_zlim(), zlim0))
    fig2, ax2 = plot_limits(fig0, ax0, zlim=(0,100))
    nt.ok_(np.allclose(ax2.get_zlim(), (0,100)))

def test_plot_ticks():
    fig1, ax1 = plot_ticks(fig, ax)
    nt.ok_(len(ax1.get_xticks()) != 0 )
    nt.ok_(len(ax1.get_yticks()) != 0 )
    fig1, ax1 = plot_ticks(fig, ax, xticks=[], yticks=[])
    nt.ok_(len(ax1.get_xticks()) == 0 )
    nt.ok_(len(ax1.get_yticks()) == 0 )
    fig1, ax1 = plot_ticks(fig, ax, xticks=np.arange(3), yticks=np.arange(4))
    nt.ok_(len(ax1.get_xticks()) == 3 )
    nt.ok_(len(ax1.get_yticks()) == 4 )
    fig2, ax2 = plot_ticks(fig0, ax0)
    nt.ok_(len(ax2.get_zticks()) != 0 )
    fig2, ax2 = plot_ticks(fig0, ax0, zticks=[])
    nt.ok_(len(ax2.get_zticks()) == 0 )
    fig2, ax2 = plot_ticks(fig0, ax0, zticks=np.arange(3))
    nt.ok_(len(ax2.get_zticks()) == 3 )

def test_plot_style():
    fig1, ax1 = plot_style(fig, ax)
    nt.ok_(ax1.get_title() == 'Figure')
    nt.ok_(ax1.get_xlabel() == 'X')
    nt.ok_(ax1.get_ylabel() == 'Y')
    fig1, ax1 = plot_style(fig, ax, no_axes=True)
    nt.ok_(ax1.get_frame_on() == False)
    nt.ok_(ax1.xaxis.get_visible() == False)
    nt.ok_(ax1.yaxis.get_visible() == False)
    fig1, ax1 = plot_style(fig, ax, tight=True)
    nt.ok_(fig1.get_tight_layout() == True)
    fig1, ax1 = plot_style(fig, ax, show_plot=False)
    nt.ok_(fig1 is None)
    nt.ok_(ax1 is None)
    if os.path.isdir(fig_dir):
        for data in os.listdir(fig_dir):
            os.remove(fig_dir + data)
        os.rmdir(fig_dir)
    fig1, ax1 = plot_style(fig, ax, output_path=fig_dir, output_name='Figure')
    nt.ok_(os.path.isfile(fig_dir + fig_name)==True)
    os.remove(fig_dir + fig_name)
    os.rmdir(fig_dir)

def test_get_color():
    nt.ok_(get_color(None, NeuriteType.basal_dendrite) == "red")
    nt.ok_(get_color(None, NeuriteType.axon) == "blue")
    nt.ok_(get_color(None, NeuriteType.apical_dendrite) == "purple")
    nt.ok_(get_color(None, NeuriteType.soma) == "black")
    nt.ok_(get_color(None, NeuriteType.undefined) == "green")
    nt.ok_(get_color(None, 'wrong') == "green")
    nt.ok_(get_color('blue', 'wrong') == "blue")
    nt.ok_(get_color('yellow', NeuriteType.axon) == "yellow")

def test_plot_sphere():
    fig0, ax0 = get_figure(params={'projection':'3d'})
    fig1, ax1 = plot_sphere(fig0, ax0, [0,0,0], 10., color='black', alpha=1.)
    nt.ok_(ax1.has_data() == True)
