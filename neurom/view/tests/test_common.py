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

def test_plot_title():
    fig1, ax1 = plot_title(fig, ax)
    nt.ok_(ax1.get_title() == 'Figure')
    fig1, ax1 = plot_title(fig, ax, title='Test')
    nt.ok_(ax1.get_title() == 'Test')
    fig1, ax1 = plot_title(fig, ax, no_title=True)
    nt.ok_(ax1.get_title() == "")
    
def test_plot_labels():
    fig1, ax1 = plot_labels(fig, ax)
    nt.ok_(ax1.get_xlabel() == 'X')
    nt.ok_(ax1.get_ylabel() == 'Y')
    fig1, ax1 = plot_labels(fig, ax, xlabel='T', no_ylabel=True)
    nt.ok_(ax1.get_xlabel() == 'T')
    nt.ok_(ax1.get_ylabel() == "")
    fig1, ax1 = plot_labels(fig, ax, ylabel='T', no_xlabel=True)
    nt.ok_(ax1.get_ylabel() == 'T')
    nt.ok_(ax1.get_xlabel() == "")
    fig1, ax1 = plot_labels(fig, ax, no_labels=True)
    nt.ok_(ax1.get_xlabel() == "")
    nt.ok_(ax1.get_ylabel() == "")

def test_plot_legend():
    fig1, ax1 = plot_legend(fig, ax)
    legend = ax1.get_legend()
    nt.ok_(legend is None)
    fig1, ax1 = plot_legend(fig, ax, no_legend=False)
    legend = ax1.get_legend()
    nt.ok_(legend.get_texts()[0].get_text() == 'test')

def test_plot_limits():
    fig1, ax1 = plot_limits(fig, ax, no_limits=True)
    nt.ok_(ax1.get_xlim() == xlim)
    nt.ok_(ax1.get_ylim() == ylim)
    fig1, ax1 = plot_limits(fig, ax, xlim=(0,100), ylim=(-100,0))
    nt.ok_(ax1.get_xlim() == (0,100))
    nt.ok_(ax1.get_ylim() == (-100,0))

def test_plot_ticks():
    fig1, ax1 = plot_ticks(fig, ax, no_ticks=False)
    nt.ok_(len(ax1.get_xticks()) != 0 )
    nt.ok_(len(ax1.get_yticks()) != 0 )
    fig1, ax1 = plot_ticks(fig, ax, no_ticks=True)
    nt.ok_(len(ax1.get_xticks()) == 0 )
    nt.ok_(len(ax1.get_yticks()) == 0 )
    fig1, ax1 = plot_ticks(fig, ax, xticks=[], yticks=[], no_xticks=True, no_yticks=True)
    nt.ok_(len(ax1.get_xticks()) == 0 )
    nt.ok_(len(ax1.get_yticks()) == 0 )
    fig1, ax1 = plot_ticks(fig, ax, xticks=np.arange(3), yticks=np.arange(4))
    nt.ok_(len(ax1.get_xticks()) == 3 )
    nt.ok_(len(ax1.get_yticks()) == 4 )

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





