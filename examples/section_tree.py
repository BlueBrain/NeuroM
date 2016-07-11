'''Example showing how to extract section information from SWC block'''
# pylint: disable=protected-access

import sys
import neurom as nm
from neurom.fst import _neuritefunc as _nf
from neurom.analysis.morphmath import section_length


def do_stuff(filename):
    '''Use the section trees to get some basic stats'''

    print '\nfst module'

    _n = nm.load_neuron(filename)

    for nt in (nm.NeuriteType.axon,
               nm.NeuriteType.basal_dendrite,
               nm.NeuriteType.apical_dendrite,
               nm.NeuriteType.all):
        print '\nNeuriteType:', nt
        n_sec = _nf.n_sections(_n, nt)
        n_seg = _nf.n_segments(_n, nt)
        sec_len = _nf.map_sections(section_length, _n, nt)

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
