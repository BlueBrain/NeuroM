'''Example showing how to extract section information from SWC block'''
# pylint: disable=protected-access

import sys
from neurom import fst
from neurom.analysis.morphmath import section_length


def do_stuff(filename):
    '''Use the section trees to get some basic stats'''

    print '\nfst module'

    _n = fst.load_neuron(filename)

    for nt in (fst.NeuriteType.axon,
               fst.NeuriteType.basal_dendrite,
               fst.NeuriteType.apical_dendrite,
               fst.NeuriteType.all):
        print '\nNeuriteType:', nt
        n_sec = fst._mm.n_sections(_n, nt)
        n_seg = fst._mm.n_segments(_n, nt)
        sec_len = fst._mm.map_sections(section_length, _n, nt)

        print 'number of sections:', n_sec
        print 'number of segments:', n_seg
        print 'total neurite length:', sum(sec_len)

    print '\nneurite types:'
    for n in _n.neurites:
        print n.type


if __name__ == '__main__':

    if len(sys.argv) < 2:
        fname = 'test_data/swc//Neuron.swc'
    else:
        fname = sys.argv[1]

    print 'loading file', fname

    do_stuff(fname)
