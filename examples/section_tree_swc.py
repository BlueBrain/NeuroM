'''Example showing how to extract section information from SWC block'''
from neurom import ezy
from neurom import _fst


def do_new_stuff(filename):
    '''Use the section trees to get some basic stats'''
    _n = _fst.load_neuron(filename)

    for nt in (ezy.NeuriteType.axon,
               ezy.NeuriteType.basal_dendrite,
               ezy.NeuriteType.apical_dendrite,
               ezy.NeuriteType.all):
        print '\nNeuriteType:', nt
        n_sec = _fst.mm.n_sections(_n, nt)
        n_seg = _fst.mm.n_segments(_n, nt)
        sec_len = _fst.mm.section_lengths(_n, nt)

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

    nrn = _fst.load_neuron(fname)
