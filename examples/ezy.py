#!/usr/bin/env python
'''Easy analysis examples

These examples highlight most of the pre-packages neurom.ezy.Neuron
morphometrics functionality.

'''

from __future__ import print_function
from pprint import pprint
from neurom import ezy
from neurom.core.types import TreeType
from neurom.core.types import NEURITES
from neurom.core.dataformat import COLS
from neurom.core import tree as tr
from neurom.analysis import morphmath as mm
import numpy as np


def stats(data):
    '''Dictionary with summary stats for data

    Returns:
        dicitonary with length, mean, sum, standard deviation,\
            min and max of data
    '''
    return {'len': len(data),
            'mean': np.mean(data),
            'sum': np.sum(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)}


def pprint_stats(data):
    '''Pretty print summary stats for data'''
    pprint(stats(data))


if __name__ == '__main__':

    filename = 'test_data/swc/Neuron.swc'

    #  load a neuron from an SWC file
    nrn = ezy.Neuron(filename)

    # Most neurite data can be queried for a particular type of neurite.
    # The allowed types are members of the TreeType enumeration.
    # NEURITES is a list of valid neurite types.

    # We start by calling methods for different neurite types separately
    # to warm up...

    # number of neurites
    print('Number of neurites (all):', nrn.get_n_neurites())
    print('Number of neurites (axons):', nrn.get_n_neurites(TreeType.axon))
    print('Number of neurites (apical dendrites):',
          nrn.get_n_neurites(TreeType.apical_dendrite))
    print('Number of neurites (basal dendrites):',
          nrn.get_n_neurites(TreeType.basal_dendrite))

    # number of sections
    print('Number of sections:', nrn.get_n_sections())
    print('Number of sections (axons):', nrn.get_n_sections(TreeType.axon))
    print('Number of sections (apical dendrites):',
          nrn.get_n_sections(TreeType.apical_dendrite))
    print('Number of sections (basal dendrites):',
          nrn.get_n_sections(TreeType.basal_dendrite))

    # number of sections per neurite
    print('Number of sections per neurite:',
          nrn.get_n_sections_per_neurite())
    print('Number of sections per neurite (axons):',
          nrn.get_n_sections_per_neurite(TreeType.axon))
    print('Number of sections per neurite (apical dendrites):',
          nrn.get_n_sections_per_neurite(TreeType.apical_dendrite))
    print('Number of sections per neurite (basal dendrites):',
          nrn.get_n_sections_per_neurite(TreeType.basal_dendrite))

    # OK, this is getting repetitive, so lets loop over valid neurite types.
    # The following methods return arrays of measurements. We will gather some
    # summary statistics for each and print them.

    # Section lengths for all and different types of neurite
    for ttype in NEURITES:
        sec_len = nrn.get_section_lengths(ttype)
        print('Section lengths (', ttype, '):', sep='')
        pprint_stats(sec_len)

    # Segment lengths for all and different types of neurite
    for ttype in NEURITES:
        seg_len = nrn.get_segment_lengths(ttype)
        print('Segment lengths (', ttype, '):', sep='')
        pprint_stats(seg_len)

    # Section radial distances for all and different types of neurite
    # Careful! Here we need to pass tree type as a named argument
    for ttype in NEURITES:
        sec_rad_dist = nrn.get_section_radial_distances(neurite_type=ttype)
        print('Section radial distance (', ttype, '):', sep='')
        pprint_stats(sec_rad_dist)

    # Section path distances for all and different types of neurite
    # Careful! Here we need to pass tree type as a named argument
    for ttype in NEURITES:
        sec_path_dist = nrn.get_section_path_distances(neurite_type=ttype)
        print('Section path distance (', ttype, '):', sep='')
        pprint_stats(sec_path_dist)

    # Local bifurcation angles for all and different types of neurite
    for ttype in NEURITES:
        local_bifangles = nrn.get_local_bifurcation_angles(ttype)
        print('Local bifurcation angles (', ttype, '):', sep='')
        pprint_stats(local_bifangles)

    # Remote bifurcation angles for all and different types of neurite
    for ttype in NEURITES:
        rem_bifangles = nrn.get_remote_bifurcation_angles(ttype)
        print('Local bifurcation angles (', ttype, '):', sep='')
        pprint_stats(rem_bifangles)

    # Now some examples of what can be done using iteration
    # instead of pre-packaged functions that return lists.
    # The iterations give us a lot of flexibility: we can map
    # any function that takes a segment or section.

    # Get length of all neurites in cell (from sections)
    # This is the most generic way:
    #   we specify type of iteration (tr.isection),
    #   and mapping function. Optionally, we can specify tree type as before.
    print('Total neurite length (sections):',
          sum(l for l in nrn.neurite_iter(tr.isection, mm.path_distance)))

    # Get length of all neurites in cell (from iter_sections)
    # Same as above, but using a more convenient section iteration
    # function instead of passing iteration mode as argument.
    print('Total neurite length (iter_sections):',
          sum(l for l in nrn.iter_sections(mm.path_distance)))

    # Get length of all neurites in cell
    # The long way, for illustration.
    # The length as calculated from segments should be the same
    # as that calculated from sections
    print('Total neurite length (segments):',
          sum(l for l in nrn.neurite_iter(tr.isegment, mm.segment_length)))

    # Get length of all neurites in cell (iter_segments)
    # Equivalent to the above, using convenience segment iteration method.
    print('Total neurite length (iter_segments):',
          sum(l for l in nrn.iter_segments(mm.segment_length)))

    # get volume of all neurites in cell
    print('Total neurite volume:',
          sum(l for l in nrn.iter_segments(mm.segment_volume)))

    # get area of all neurites in cell
    print('Total neurite surface area:',
          sum(l for l in nrn.iter_segments(mm.segment_area)))

    # get total number of points in cell
    print('Total number of points:',
          sum(1 for _ in nrn.iter_points(lambda p: p)))

    # get mean radius of points in cell
    print('Mean radius of points:',
          np.mean([r for r in nrn.iter_points(lambda p: p[COLS.R])]))

    # get mean radius of segments
    print('Mean radius of segments:',
          np.mean([r for r in nrn.iter_segments(mm.segment_radius)]))
