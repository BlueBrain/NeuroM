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

    # Get some soma information
    # Soma radius and surface area
    print("Soma radius", nrn.get_soma_radius())
    print("Soma surface area", nrn.get_soma_surface_area())

    # Get information about neurites
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

    # Get length of all neurites in cell by iterating over sections,
    # and summing the section lengths
    print('Total neurite length:',
          sum(seclen for seclen in nrn.iter_sections(mm.path_distance)))

    # Get length of all neurites in cell by iterating over segments,
    # and summing the segment lengths.
    # This should yield the same result as iterating over sections.
    print('Total neurite length:',
          sum(seglen for seglen in nrn.iter_segments(mm.segment_length)))

    # get volume of all neurites in cell by summing over segment
    # volumes
    print('Total neurite volume:',
          sum(vol for vol in nrn.iter_segments(mm.segment_volume)))

    # get area of all neurites in cell by summing over segment
    # areas
    print('Total neurite surface area:',
          sum(area for area in nrn.iter_segments(mm.segment_area)))

    # get total number of points in cell.
    # iter_points needs a mapping function, so we pass the identity.
    print('Total number of points:',
          sum(1 for _ in nrn.iter_points(lambda p: p)))

    # get mean radius of points in cell.
    # p[COLS.R] yields the radius for point p.
    print('Mean radius of points:',
          np.mean([r for r in nrn.iter_points(lambda p: p[COLS.R])]))

    # get mean radius of segments
    print('Mean radius of segments:',
          np.mean([r for r in nrn.iter_segments(mm.segment_radius)]))
