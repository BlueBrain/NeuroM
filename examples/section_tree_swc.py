'''Example showing how to extract section information from SWC block'''
from neurom import ezy
from neurom.io import swc
from neurom.core import section_neuron as sn
from neurom.core.dataformat import COLS
from neurom.core.dataformat import POINT_TYPE


def remove_soma_initial_point(tree, post_action=sn.set_neurite_type):
    '''Remove tree's initial point if soma and apply post_action'''
    if tree.value[0][COLS.TYPE] == POINT_TYPE.SOMA:
        tree.value = tree.value[1:]

    if post_action is not None:
        post_action(tree)


def load_neuron(filename, tree_action=remove_soma_initial_point):
    '''Build section trees from an swc file'''
    rdw = swc.SWC.read(filename, sn.SecDataWrapper)
    trees = sn.make_trees(rdw, tree_action)
    soma = sn.make_soma(rdw.soma_points())
    return sn.Neuron(soma, trees, rdw)


def do_new_stuff(filename):
    '''Use the section trees to get some basic stats'''
    _n = sn.load_neuron(filename)

    for nt in (sn.NeuriteType.axon,
               sn.NeuriteType.basal_dendrite,
               sn.NeuriteType.apical_dendrite,
               sn.NeuriteType.all):
        print '\nNeuriteType:', nt
        n_sec = sn.n_sections(_n, nt)
        n_seg = sn.n_segments(_n, nt)
        sec_len = sn.get_section_lengths(_n, nt)

        print 'number of sections:', n_sec
        print 'number of segments:', n_seg
        print 'total neurite length:', sum(sec_len)

    print '\nneurite types:'
    for n in _n.neurites:
        print n.type


def do_old_stuff(filename):
    '''Use point tree to get some basic stats'''
    _n = ezy.load_neuron(filename)

    for nt in (sn.NeuriteType.axon,
               sn.NeuriteType.basal_dendrite,
               sn.NeuriteType.apical_dendrite,
               sn.NeuriteType.all):
        print '\nNeuriteType:', nt
        n_sec = ezy.get('number_of_sections', _n, neurite_type=nt)[0]
        n_seg = ezy.get('number_of_segments', _n, neurite_type=nt)[0]
        sec_len = ezy.get('section_lengths', _n, neurite_type=nt)

        print 'number of sections:', n_sec
        print 'number of segments:', n_seg
        print 'total neurite length:', sum(sec_len)

    print '\nneurite types:'
    for n in _n.neurites:
        print n.type

if __name__ == '__main__':

    fname = 'test_data/swc/Neuron.swc'

    nrn = sn.load_neuron(fname)
