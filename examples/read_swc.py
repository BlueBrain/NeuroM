from collections import defaultdict
import numpy as np

# SWC n, T, x, y, z, R, P
(ID, TYPE, X, Y, Z, R, P) = xrange(7)


class SWCData(object):
    def __init__(self, filename):
        self.data_block = np.loadtxt(filename)
        self.adj_list = defaultdict(list)
        self.offset_ = self.data_block[0][0]
        for row in self.data_block:
            self.adj_list[int(row[P])].append(int(row[ID]))


    def get_children(self, idx):
        return self.adj_list[idx]


    def get_parent(self, idx):
        return int(self.data_block[idx - self.offset_][P])


    def get_endpoints(self):
        ''' get the end points of the tree

        End points have no children so are not in the
        adjacency list.
        '''
        return set(self.data_block[:,ID]) - set(self.adj_list.keys())


    def get_bifurcation_points(self):
        return [i for i, l in self.adj_list.iteritems() if len(l) > 1]
