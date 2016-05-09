'''Example showing how to extract section information from SWC block'''
import numpy as np
from neurom import ezy
from neurom.io import swc
from neurom.core.tree import Tree
from neurom.core import section_neuron as sn
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE


class Section(object):
    '''sections (id, (ids), type, parent_id)'''
    def __init__(self, idx, ids=None, ntype=0, pid=-1):
        self.id = idx
        self.ids = [] if ids is None else ids
        self.ntype = ntype
        self.pid = pid

    def __str__(self):
        return 'Section(id=%s, ids=%s, ntype=%s, pid=%s)' % (self.id, self.ids,
                                                             self.ntype, self.pid)


def neurite_trunks(data_wrapper):
    '''Get the section IDs of the intitial neurite sections'''
    sec = data_wrapper.sections
    return [ss.id for ss in sec
            if ss.pid is not None and (sec[ss.pid].ntype == POINT_TYPE.SOMA and
                                       ss.ntype != POINT_TYPE.SOMA)]


def soma_points(data_wrapper):
    '''Get the soma points'''
    db = data_wrapper.data_block
    return db[db[:, COLS.TYPE] == POINT_TYPE.SOMA]


def add_sections(data_wrapper):
    '''Make a list of sections from an SWC data wrapper'''

    # get SWC ID to array position map
    id_map = {-1: -1}
    for i, r in enumerate(data_wrapper.data_block):
        id_map[int(r[COLS.ID])] = i

    fork_points = set(id_map[p] for p in data_wrapper.get_fork_points())
    end_points = set(id_map[p] for p in data_wrapper.get_end_points())
    section_end_points = fork_points | end_points

    _sections = [Section(0)]
    curr_section = _sections[-1]
    parent_section = {-1: None}

    for row in data_wrapper.data_block:
        row_id = id_map[int(row[COLS.ID])]
        if len(curr_section.ids) == 0:
            curr_section.ids.append(id_map[int(row[COLS.P])])
            curr_section.ntype = int(row[COLS.TYPE])
        curr_section.ids.append(row_id)
        if row_id in section_end_points:
            parent_section[curr_section.ids[-1]] = curr_section.id
            _sections.append(Section(len(_sections)))
            curr_section = _sections[-1]

    # get the section parent ID from the id of the first point.
    for sec in _sections:
        if sec.ids:
            sec.pid = parent_section[sec.ids[0]]

    data_wrapper.sections = [s for s in _sections if s.ids]
    return data_wrapper


def make_tree(data_wrapper, start_node=0, post_action=None):
    '''Build a section tree'''
    # One pass over sections to build nodes
    nodes = [Tree(np.array(data_wrapper.data_block[sec.ids]))
             for sec in data_wrapper.sections[start_node:]]

    # One pass over nodes to connect children to parents
    for i in xrange(len(nodes)):
        parent_id = data_wrapper.sections[i + start_node].pid - start_node
        if parent_id >= 0:
            nodes[parent_id].add_child(nodes[i])

    if post_action is not None:
        post_action(nodes[0])

    return nodes[0]


def load_neuron(filename, tree_action=sn.set_neurite_type):
    '''Build section trees from an h5 file'''
    data_wrapper = swc.SWC.read(filename)
    add_sections(data_wrapper)
    trunks = neurite_trunks(data_wrapper)
    trees = [make_tree(data_wrapper, trunk, tree_action)
             for trunk in trunks]
    # if any neurite trunk starting points are soma,
    # remove them
    for t in trees:
        if t.value[0][COLS.TYPE] == POINT_TYPE.SOMA:
            t.value = t.value[1:]
    soma = sn.make_soma(soma_points(data_wrapper))
    return sn.Neuron(soma, trees, data_wrapper)


def do_new_stuff(filename):
    '''Use the section trees to get some basic stats'''
    _n = load_neuron(filename)

    n_sec = sn.n_sections(_n)
    n_seg = sn.n_segments(_n)
    sec_len = sn.get_section_lengths(_n)

    print 'number of sections:', n_sec
    print 'number of segments:', n_seg
    print 'total neurite length:', sum(sec_len)
    print 'neurite types:'
    for n in _n.neurites:
        print n.type


def do_old_stuff(filename):
    '''Use point tree to get some basic stats'''
    _n = ezy.load_neuron(filename)
    n_sec = ezy.get('number_of_sections', _n)[0]
    n_seg = ezy.get('number_of_segments', _n)[0]
    sec_len = ezy.get('section_lengths', _n)

    print 'number of sections:', n_sec
    print 'number of segments:', n_seg
    print 'total neurite length:', sum(sec_len)
    print 'neurite types:'
    for n in _n.neurites:
        print n.type

if __name__ == '__main__':

    fname = 'test_data/swc/Neuron.swc'

    nrn = load_neuron(fname)
