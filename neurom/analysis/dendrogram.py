

from neurom.core.tree import Tree, val_iter
from neurom.core.neuron import Neuron
from neurom.analysis.morphtree import n_segments, n_bifurcations, n_terminations
from neurom.analysis.morphmath import segment_length
from neurom.core.dataformat import COLS

from neurom.view import common
from matplotlib.collections import PolyCollection

import numpy as np
import sys


def _total_lines(tree): 

    return n_segments(tree) + n_bifurcations(tree) * 2


def _n_lines(obj):

    if isinstance(obj, Tree):

        return _total_lines(obj)

    elif isinstance(obj, Neuron):

        return sum([_total_lines(neu) for neu in obj.neurites])

    else:

        return 0


class Dendrogram(object):


    def __init__(self, obj):

        self._obj = obj
        self._n = 0

        self._max_dims = [0., 0.]
        self._offsets = [0., 0.]

        self._trees = []
        self._dims = []

        self._lines = np.zeros([_n_lines(self._obj), 4, 2])

        print "nlines : ", _n_lines(self._obj)

    def run(self):

        sys.setrecursionlimit(100000)

        spacing = (40., 0.)

        n_previous = 0
        #for neurite in self._obj.neurites if isinstance(self._obj, Neuron) else self._obj:
        if isinstance(self._obj, Tree):

            self._generate_dendro(self._obj, self._lines, spacing)

            self._trees = [self._lines]

        else:

            for neurite in self._obj.neurites:

                n_previous = 0

                self._generate_dendro(neurite, self._lines, spacing)

                # store in trees the sliced array of lines for each neurite
                self._trees.append(self._lines[n_previous : self._n])

                # store the max dims per neurite for view positioning
                self._dims.append(self._max_dims)

                # reset the max dimensions for the next tree in line
                self._max_dims = [0., 0.]
                self._offsets = [0., 0.]

                # keep track of the next tree start index in list
                n_previous = self._n



    @property
    def tr():
        return _tree


    @property
    def data():
        return _lines


    def _vertical_segment(self, old_offs, new_offs, spacing, radii):
        '''Vertices fo a vertical segment
        '''
        return np.array(((new_offs[0] - radii[0], old_offs[1] + spacing[1]),
                         (new_offs[0] - radii[1], new_offs[1]),
                         (new_offs[0] + radii[1], new_offs[1]),
                         (new_offs[0] + radii[0], old_offs[1] + spacing[1])))


    def _horizontal_segment(self, old_offs, new_offs, spacing, diameter):
        '''Vertices of a horizontal segmen
        '''
        return np.array(((old_offs[0], old_offs[1] + spacing[1]),
                         (new_offs[0], old_offs[1] + spacing[1]),
                         (new_offs[0], old_offs[1] + spacing[1] - diameter),
                         (old_offs[0], old_offs[1] + spacing[1] - diameter)))


    def _spacingx(self, node, max_dims, offsets, spacing):
        '''Determine the spacing of the current node depending on the number
           of the leaves of the tree
        '''
        x_spacing = n_terminations(node) * spacing[0]

        if x_spacing > max_dims[0]:
            max_dims[0] = x_spacing

        return offsets[0] - x_spacing / 2.

    def _update_offsets(self, start_x, spacing, terminations, offsets, length):

        return [start_x + spacing[0] * terminations / 2.,
                offsets[1] + spacing[1] * 2. + length]

    def _generate_dendro(self, current_node, lines, spacing):
        '''Recursive function for dendrogram line computations
        '''
        offsets = self._offsets
        max_dims = self._max_dims
        start_x = self._spacingx(current_node, max_dims, offsets, spacing)

        radii = [0., 0.]
        # store the parent radius in order to construct polygonal segments
        # isntead of simple line segments
        radii[0] = current_node.value[COLS.R]

        for child in current_node.children:

            # segment length
            ln = segment_length(list(val_iter((current_node, child))))

            # extract the radius of the child node. Need both radius for
            # realistic segment representation
            radii[1] = child.value[COLS.R]

            # number of leaves in child
            terminations = n_terminations(child)

            # horizontal spacing with respect to the number of
            # terminations
            new_offsets = self._update_offsets(start_x, spacing, terminations, offsets, ln)

            # create and store vertical segment
            lines[self._n] = self._vertical_segment(offsets, new_offsets, spacing, radii)

            # assign segment id to color array
            #colors[n[0]] = child.value[4]
            self._n += 1

            if offsets[1] + spacing[1] * 2 + ln > max_dims[1]:
                max_dims[1] = offsets[1] + spacing[1] * 2. + ln

            self._offsets = new_offsets
            self._max_dims = max_dims
            # recursive call to self.
            self._generate_dendro(child, lines, spacing)

            # update the starting position for the next child
            start_x += terminations * spacing[0]

            # write the horizontal lines only for bifurcations, where the are actual horizontal lines
            # and not zero ones
            if offsets[0] != new_offsets[0]:

                # horizontal segment
                lines[self._n] = self._horizontal_segment(offsets, new_offsets, spacing, 0.)
                #colors[self._n] = current_node.value[4]
                self._n += 1

    def displace(self, rectangles, t):
        print
        
        for rectangle in rectangles:

            
            for line in rectangle:


                line[0] += t[0]
                line[1] += t[1]

    def view(self, new_fig=True, subplot=None, **kwargs):


        fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

        dist = 0.
        for i, group in enumerate(self._trees):
            print i
            #if i > 0:
            # dist += 0.5 * (self._dims[i-1][0] + self._dims[i][0])
            #print dist
            #print
            dist += 10.
            #print dist
            #self.displace(group, [dist, 0.])
            collection = PolyCollection(group, closed=False, antialiaseds=True, offsets=(0., dist))
            #return collection
            #if i == 0:
            #    dist += 0.5 * self._dims[i][0]
            ax.add_collection(collection)

        ax.autoscale(enable=True, tight=None)

        # dummy plots for color bar labels
        #for color, label in set(linked_colors):
        #    ax.plot((0., 0.), (0., 0.), c=color, label=label)

        # customization settings
        #kwargs['xticks'] = []
        #kwargs['title'] = kwargs.get('title', 'Morphology Dendrogram')
        #kwargs['xlabel'] = kwargs.get('xlabel', '')
        #kwargs['ylabel'] = kwargs.get('ylabel', '')
        #kwargs['no_legend'] = False

        return common.plot_style(fig=fig, ax=ax, **kwargs)

