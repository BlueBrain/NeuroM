

from neurom.core.tree import Tree, val_iter
from neurom.core.neuron import Neuron
from neurom.analysis.morphtree import n_segments, n_bifurcations, n_terminations
from neurom.analysis.morphmath import segment_length
from neurom.core.dataformat import COLS

from neurom.view import common
from matplotlib.collections import PolyCollection

import numpy as np


def _total_lines(tree): 

    return n_segments(tree) + n_bifurcations(tree) * 2


def _n_lines(obj):

    if isinstance(obj, Tree):

        return _total_lines(tree)

    elif isinstance(obj, Neuron):

        return sum([_total_lines(neu) for neu in obj.neurites])

    else:

        return 0


class Dendrogram(object):


    def __init__(self, obj):

        self._tree = None
        self._n = 0

        self._max_dims = [0., 0.]
        self._spacing = (40., 0.)
        self._offsets = [0., 0.]

        self._neurites = []

        self._lines = np.zeros([_n_lines(obj), 4, 2])

    def run(self, obj):

        n_previous = 0
        for neurite in obj.neurites:

            self._generate_dendro(neurite)

            self._neurites.append(self._lines[n_previous : self._n])

            n_previous = self._n


    @property
    def tr():
        return _tree


    @property
    def data():
        return _lines


    def _vertical_segment(self, new_offsets, radii):
        '''Vertices fo a vertical segment
        '''
        return np.array(((new_offsets[0] - radii[0], self._offsets[1] + self._spacing[1]),
                         (new_offsets[0] - radii[1], new_offsets[1]),
                         (new_offsets[0] + radii[1], new_offsets[1]),
                         (new_offsets[0] + radii[0], self._offsets[1] + self._spacing[1])))


    def _horizontal_segment(self, new_offsets, diameter):
        '''Vertices of a horizontal segmen
        '''
        return np.array(((self._offsets[0], self._offsets[1] + self._spacing[1]),
                         (new_offsets[0], self._offsets[1] + self._spacing[1]),
                         (new_offsets[0], self._offsets[1] + self._spacing[1] - diameter),
                         (self._offsets[0], self._offsets[1] + self._spacing[1] - diameter)))


    def _spacingx(self, node):
        '''Determine the spacing of the current node depending on the number
           of the leaves of the tree
        '''
        x_spacing = n_terminations(node) * self._spacing[0]

        if x_spacing > self._max_dims[0]:
            self._max_dims[0] = x_spacing

        return self._offsets[0] - x_spacing / 2.


    def _generate_dendro(self, current_node):
        '''Recursive function for dendrogram line computations
        '''

        start_x = self._spacingx(current_node)

        radii = [0., 0.]
        # store the parent radius in order to construct polygonal segments
        # isntead of simple line segments
        radii[0] = current_node.value[COLS.R]

        for child in current_node.children:

            # segment length
            length = segment_length(list(val_iter((current_node, child))))

            # extract the radius of the child node. Need both radius for
            # realistic segment representation
            radii[1] = child.value[COLS.R]

            # number of leaves in child
            terminations = n_terminations(child)

            # horizontal spacing with respect to the number of
            # terminations
            new_offsets = (start_x + self._spacing[0] * terminations / 2.,
                           self._offsets[1] + self._spacing[1] * 2. + length)

            # vertical segment
            self._lines[self._n] = self._vertical_segment(new_offsets, radii)

            # assign segment id to color array
            #self.colors[self._n] = child.value[COLS.TYPE]

            self._n += 1

            if self._offsets[1] + self._spacing[1] * 2 + length > self._max_dims[1]:
                self._max_dims[1] = self._offsets[1] + self._spacing[1] * 2. + length

            # recursive call to self.
            self._generate_dendro(child)

            # update the starting position for the next child
            start_x += terminations * self._spacing[0]

            # write the horizontal lines only for bifurcations, where the are actual horizontal lines
            # and not zero ones
            if self._offsets[0] != new_offsets[0]:

                # horizontal segment
                self._lines[self._n] = self._horizontal_segment(new_offsets, 0.)
                #colors[self._n] = current_node.value[COLS.TYPE]
                self._n += 1

    def view(self, new_fig=True, subplot=None):


        collection = PolyCollection(self._lines, closed=False, antialiaseds=True)

        fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

        ax.add_collection(collection)
        ax.autoscale(enable=True, tight=None)

        # dummy plots for color bar labels
        #for color, label in set(linked_colors):
        #    ax.plot((0., 0.), (0., 0.), c=color, label=label)

        # customization settings
        kwargs['xticks'] = []
        kwargs['title'] = kwargs.get('title', 'Morphology Dendrogram')
        kwargs['xlabel'] = kwargs.get('xlabel', '')
        kwargs['ylabel'] = kwargs.get('ylabel', '')
        kwargs['no_legend'] = False

        return common.plot_style(fig=fig, ax=ax, **kwargs)

