from neurom.features import NEURITEFEATURES as nf
import pylab as plt
from neurom.view import common
from neurom.view import view

def extract_density(population, plane='xy', bins=100):

    horiz = nf['segment_' + plane[0] + '_coordinates'](pop)
    vert = nf['segment_' + plane[1] + '_coordinates'](pop)

    return np.histogram2d(np.array(horiz), np.array(vert),
                          bins=(bins, bins))


def plot_density(population, plane='xy', bins=100, new_fig=True, subplot=111,
                 colorlabel='Nodes per unit area', labelfontsize=16, levels=None,
                 color_map='Reds', no_colorbar=False, threshold=0.01, **kwargs):

    fig, ax = fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    H1, xedges1, yedges1 = extract_density(population, plane=plane, bins=bins)

    mask = H1 < threshold  # mask = H1==0
    H2 = np.ma.masked_array(H1, mask)

    getattr(plt.cm, color_map).set_bad(color='white', alpha=None)

    plots = ax.contourf((xedges1[:-1] + xedges1[1:]) / 2,
                        (yedges1[:-1] + yedges1[1:]) / 2,
                        np.transpose(H2), # / np.max(H2),
                        cmap=getattr(plt.cm, color_map), levels=levels)

    if not no_colorbar:
        cbar = plt.colorbar(plots)
        cbar.ax.set_ylabel(colorlabel, fontsize=labelfontsize)

    kwargs['title'] = kwargs.get('title', '')
    kwargs['xlabel'] = kwargs.get('xlabel', plane[0])
    kwargs['ylabel'] = kwargs.get('ylabel', plane[1])

    return common.plot_style(fig=fig, ax=ax, **kwargs)


def plot_neuron_on_density(population, plane='xy', bins=100, new_fig=True, subplot=111,
                           colorlabel='Nodes per unit area', labelfontsize=16, levels=None,
                           color_map='Reds', no_colorbar=False, threshold=0.01, **kwargs):

    fig, ax = view.tree(pop.neurites[0], new_fig=new_fig)
    
    return plot_density(population, plane=plane, bins=bins, new_fig=False, subplot=subplot,
                        colorlabel=colorlabel, labelfontsize=labelfontsize, levels=levels,
                        color_map=color_map, no_colorbar=no_colorbar, threshold=threshold, **kwargs)
