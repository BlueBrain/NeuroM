#!/usr/bin/env python
'''example for finding the apical point, and potentially plotting it'''
import argparse
import logging

import neurom as nm
from neurom.core.dataformat import COLS
from neurom.morphmath import point_dist2

L = logging.getLogger(__name__)


def get_apical_point(morph, tuft_percent=20):
    '''Attempt to find the apical point in 'tufted' neurons

    Consider a neuron:

        |   /    | Tuft = 20%
        |--/     |
        |   /
        |--/
        |
    ----.-----

    All endpoints in the top 'tuft_percent' are found, then their common
    branch segment, furthest from the soma, is identified.

    Args:
        morph: neurom.fst._core.Neuron
        tuft_percent: percentage of the 'height' of the apical dendrite that
        would enclose the tuft, only leaves in this volume are considered as
        endpoints.  Note that this a spherical shell centered at the soma

    Returns:
        Section whose *end* is where the apical branching begins, or None if there
        is a problem
    '''
    apical = [neurite for neurite in morph.neurites
              if nm.NeuriteType.apical_dendrite == neurite.type]

    if not apical:
        L.warning('No apical found')
        return None
    elif 1 < len(apical):
        L.warning('Too many apical dendrites')
        return None

    apical = apical[0]

    max_distance2 = -float('inf')
    for leaf in apical.root_node.ileaf():
        point = leaf.points[-1, COLS.XYZ]
        max_distance2 = max(max_distance2,
                            point_dist2(point, morph.soma.center))

    min_distance2 = max_distance2 * (1 - tuft_percent / 100.) ** 2

    common_parents = set(nm.iter_sections(apical))
    for leaf in apical.root_node.ileaf():
        point = leaf.points[-1, COLS.XYZ]
        if min_distance2 <= point_dist2(point, morph.soma.center):
            parents = leaf.iupstream()
            common_parents &= set(parents)

    for parent_section in nm.iter_sections(apical):
        if parent_section in common_parents:
            common_parents.remove(parent_section)
            if not common_parents:
                return parent_section

    return None


def get_parser():
    '''return the argument parser'''
    parser = argparse.ArgumentParser()
    parser.add_argument('morph',
                        help='morphology to find the apical point')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot the morphology with the apical point')
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='-v for INFO, -vv for DEBUG')
    return parser


def main(args):
    '''main function'''

    if args.verbose > 2:
        raise Exception('cannot be more verbose than -vv')

    logging.basicConfig(level=(logging.WARNING,
                               logging.INFO,
                               logging.DEBUG)[args.verbose])

    morph = nm.load_neuron(args.morph)
    apical_point = get_apical_point(morph)
    pos = apical_point.points[-1, COLS.XYZ]

    if args.plot:
        import matplotlib.pyplot as plt
        from neurom.viewer import draw
        _, ax = draw(morph)
        ax.scatter([pos[0], ], [pos[1], ])
        plt.show()

    print('Apical point is in section id: %s, at %s' % (apical_point.id, pos))

if __name__ == '__main__':
    PARSER = get_parser()
    main(PARSER.parse_args())
