'''Build a section tree'''

from collections import defaultdict
from collections import namedtuple
import sys
import numpy as np
from neurom.io.hdf5 import H5
from neurom.core.tree import Tree, ipreorder
from neurom.core.dataformat import POINT_TYPE
from neurom.core.tree import i_chain2, iupstream
from neurom.analysis import morphmath as mm
from neurom import ezy


(START, END, TYPE, ID, PID) = xrange(5)


MockNeuron = namedtuple('MockNeuron', 'neurites, data_block, adj_list')
MockWrapper = namedtuple('MockWrapper', 'data_block, fmt, sections')


def buid_adjacency_list(rdw):
    '''Build an adjacency list of sections'''
    adj_list = defaultdict(list)

    for sec in rdw.sections:
        adj_list[sec[PID]].append(sec[ID])

    return adj_list


def make_tree(rdw, adj_list, root_node=0):
    '''Build a section tree'''
    _sec = rdw.sections
    head_node = Tree(_sec[root_node])
    children = [head_node]
    while children:
        cur_node = children.pop()
        for c in adj_list[cur_node.value[ID]]:
            child = Tree(_sec[c])
            cur_node.add_child(child)
            children.append(child)

    # set the data in each node to a slice of the raw data block
    for n in ipreorder(head_node):
        sec = n.value
        n.value = rdw.data_block[sec[START]: sec[END]]

    return head_node


def get_neurite_sections(rdw):
    '''Get the section IDs of the intitial neurite sections'''
    sec = rdw.sections
    return [ss[ID] for ss in sec if sec[ss[PID]][TYPE] == POINT_TYPE.SOMA]


def load_nrn(filename):
    '''Build section trees from an h5 file'''
    # unpack the data
    rdw = H5.read(filename, remove_duplicates=False, wrapper=MockWrapper)
    # get the adjacency list
    adj_list = buid_adjacency_list(rdw)

    # get the initial neurite sections
    trunks = get_neurite_sections(rdw)
    trees = [make_tree(rdw, adj_list, trunk) for trunk in trunks]
    nrn = MockNeuron(trees, rdw, adj_list)

    return nrn


def n_segments(section):
    '''Number of segments in a section'''
    return len(section.value) - 1


def path_length(section):
    '''Path length from section to root'''
    return sum(mm.path_distance(s.value) for s in iupstream(section))


def do_new_stuff(filename):
    '''Use the section trees to get some basic stats'''
    _n = load_nrn(filename)

    n_sec = sum(1 for _ in i_chain2(_n.neurites))
    n_seg = sum(n_segments(s) for s in i_chain2(_n.neurites))
    sec_len = [mm.path_distance(s.value) for s in i_chain2(_n.neurites)]

    print 'number of sections:', n_sec
    print 'number of segments:', n_seg
    print 'tota neurite length:', sum(sec_len)


def get_path_lengths_new(nrn):
    '''Use new stuff to get path length of leaf neurites'''
    return np.array([path_length(s) for s in i_chain2(nrn.neurites)])


def get_path_lengths_new2(nrn):
    '''Less naive path length calculation

    Calculates and stores the section lengths in one pass,
    then queries the lengths in the path length iterations.
    This avoids repeatedly calculating the lengths of the
    same sections.
    '''
    dist = {}

    for s in i_chain2(nrn.neurites):
        dist[s] = mm.path_distance(s.value)

    def pl2(sec):
        '''Calculate the path length using cahced section lengths'''
        return sum(dist[s] for s in iupstream(sec))

    return np.array([pl2(s) for s in i_chain2(nrn.neurites)])


def get_path_lengths_old(nrn):
    '''old school'''
    return ezy.get('section_path_distances', nrn)


def do_old_stuff(filename):
    '''Use point tree to get some basic stats'''
    _n = ezy.load_neuron(filename)
    n_sec = ezy.get('number_of_sections', _n)[0]
    n_seg = ezy.get('number_of_segments', _n)[0]
    sec_len = ezy.get('section_lengths', _n)

    print 'number of sections:', n_sec
    print 'number of segments:', n_seg
    print 'tota neurite length:', sum(sec_len)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        fname = 'test_data/h5/v1/Neuron_2_branch.h5'
    else:
        fname = sys.argv[1]

    print 'loading file', fname
