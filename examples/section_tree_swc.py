'''Example showing how to extract section information from SWC block'''
# pylint: disable=protected-access

from neurom import ezy
from neurom import fst


def do_new_stuff(filename):
    '''Use the section trees to get some basic stats'''
    _n = fst.load_neuron(filename)

    for nt in (ezy.NeuriteType.axon,
               ezy.NeuriteType.basal_dendrite,
               ezy.NeuriteType.apical_dendrite,
               ezy.NeuriteType.all):
        print '\nNeuriteType:', nt
        n_sec = fst._mm.n_sections(_n, nt)
        n_seg = fst._mm.n_segments(_n, nt)
        sec_len = fst._mm.section_lengths(_n, nt)

        print 'number of sections:', n_sec
        print 'number of segments:', n_seg
        print 'total neurite length:', sum(sec_len)

    print '\nneurite types:'
    for n in _n.neurites:
        print n.type


def do_old_stuff(filename):
    '''Use point tree to get some basic stats'''
    _n = ezy.load_neuron(filename)

    for nt in (ezy.NeuriteType.axon,
               ezy.NeuriteType.basal_dendrite,
               ezy.NeuriteType.apical_dendrite,
               ezy.NeuriteType.all):
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

    nrn = fst.load_neuron(fname)
