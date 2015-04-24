from nose import tools as nt
from neurom.view.common import figure_naming
from neurom.view.common import get_figure
from neurom.view.common import save_plot
from neurom.view.common import style_plot
import matplotlib.pyplot as plt 
import os

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
    fig_name = 'Figure.png'
    fig_dir  = './Testing_save_in_directory/'
    if os.path.isfile(fig_name):
        os.remove(fig_name)
    fig_old = plt.figure()
    save_plot(fig_old)
    nt.ok_(os.path.isfile(fig_name)==True)
    os.remove(fig_name)
    if os.path.isdir(fig_dir):
        for data in os.listdir(fig_dir):
            os.remove(fig_dir + data)
        os.rmdir(fig_dir)
    save_plot(fig_old, output_path=fig_dir)
    nt.ok_(os.path.isfile(fig_dir + fig_name)==True)
    os.remove(fig_dir + fig_name)
    os.rmdir(fig_dir)
    plt.close('all')

def test_style_plot():
    fig_name = 'Figure.png'
    fig_dir  = './Testing_save_in_directory/'
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    fig0, ax0 = style_plot(fig, ax, title='Test', xlabel='T')
    nt.ok_(ax0.get_title() == 'Test')
    nt.ok_(ax0.get_xlabel() == 'T')
    nt.ok_(ax0.get_ylabel() == 'Y')
    nt.ok_(ax0.get_legend() is None)
    fig1, ax1 = style_plot(fig, ax, no_title=True, no_labels=True, no_ticks=True)
    nt.ok_(ax1.get_title() == '')
    nt.ok_(ax1.get_xlabel() == '')
    nt.ok_(ax1.get_ylabel() == '')
    nt.ok_(len(ax1.get_xticks()) == 0 )
    nt.ok_(len(ax1.get_yticks()) == 0 )
    ax.plot([0,0], [1,2], label='test')
    fig2, ax2 = style_plot(fig, ax, no_legend=False, tight=True, no_axes=True)
    legend = ax2.get_legend()
    nt.ok_(legend.get_texts()[0].get_text() == 'test')
    nt.ok_(fig2.get_tight_layout() == True)
    nt.ok_(ax2.get_frame_on() == False)
    nt.ok_(ax2.xaxis.get_visible() == False)
    nt.ok_(ax2.yaxis.get_visible() == False)
    plt.close('all')
    if os.path.isdir(fig_dir):
        for data in os.listdir(fig_dir):
            os.remove(fig_dir + data)
        os.rmdir(fig_dir)
    fig3, ax3 = style_plot(fig, ax, show_plot=False, output_path=fig_dir, output_name='Figure')
    nt.ok_(len(plt.get_fignums()) == 0)
    nt.ok_(os.path.isfile(fig_dir + fig_name)==True)
    os.remove(fig_dir + fig_name)
    os.rmdir(fig_dir)
    plt.close('all')






