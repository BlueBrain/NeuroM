'''Build a section tree'''

import sys
from neurom.core import section_neuron as sn
from neurom import ezy


def do_new_stuff(filename):
    '''Use the section trees to get some basic stats'''
    _n = sn.load_neuron(filename)

    n_sec = sn.n_sections(_n)
    n_seg = sn.n_segments(_n)
    sec_len = sn.get_section_lengths(_n)

    print 'number of sections:', n_sec
    print 'number of segments:', n_seg
    print 'tota neurite length:', sum(sec_len)


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
