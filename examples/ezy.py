#!/usr/bin/env python
'''Easy analysis examples'''

from neurom import ezy
from neurom.core import tree as tr
from neurom.analysis import morphmath as mm
import numpy as np


if __name__ == '__main__':

    filename = 'test_data/swc/Neuron.swc'

    nrn = ezy.Neuron(filename)

    sec_len_axon1 = nrn.get_section_lengths(ezy.TreeType.axon)

    sec_len_axon2 = nrn.neurite_loop(tr.isection,
                                     mm.path_distance,
                                     ezy.TreeType.axon)

    isec_len_axon = nrn.neurite_iter(tr.isection,
                                     mm.path_distance,
                                     ezy.TreeType.axon)

    iseg_len_axon = nrn.neurite_iter(tr.isegment,
                                     mm.segment_length,
                                     ezy.TreeType.axon)

    print np.allclose(sec_len_axon1, sec_len_axon2)

    print 'Total axon length from section array:', sum(sec_len_axon1)
    print 'Total axon length from sections:', sum(l for l in isec_len_axon)
    print 'Total axon length from segments:', sum(l for l in iseg_len_axon)

    # get length of all neurites in cell (from sections)
    print 'Total neurite length (sections):', sum(l for l in nrn.neurite_iter(tr.isection,
                                                                              mm.path_distance))

    # get length of all neurites in cell
    print 'Total neurite length (segments):', sum(l for l in nrn.neurite_iter(tr.isegment,
                                                                              mm.segment_length))

    # get volume of all neurites in cell
    print 'Total neurite volume:', sum(l for l in nrn.neurite_iter(tr.isegment,
                                                                   mm.segment_volume))

    # get area of all neurites in cell
    print 'Total neurite surface area:', sum(l for l in nrn.neurite_iter(tr.isegment,
                                                                         mm.segment_area))
