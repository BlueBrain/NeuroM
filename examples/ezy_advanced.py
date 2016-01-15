#!/usr/bin/env python
# Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
# All rights reserved.
#
# This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the names of
#        its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''Advanced easy analysis examples

These examples highlight more advanced neurom.ezy.Neuron
morphometrics functionality.

'''

from __future__ import print_function
from itertools import imap
from neurom import segments as seg
from neurom import sections as sec
from neurom import bifurcations as bif
from neurom import iter_neurites
from neurom import ezy
from neurom.core.types import tree_type_checker
from neurom.core.tree import isection
from neurom.core.dataformat import COLS
from neurom.analysis import morphtree as mt
import numpy as np


if __name__ == '__main__':

    filename = 'test_data/swc/Neuron.swc'

    #  load a neuron from an SWC file
    nrn = ezy.load_neuron(filename)

    # Some examples of what can be done using iteration
    # instead of pre-packaged functions that return lists.
    # The iterations give us a lot of flexibility: we can map
    # any function that takes a segment or section.

    # Get length of all neurites in cell by iterating over sections,
    # and summing the section lengths
    print('Total neurite length:',
          sum(iter_neurites(nrn, sec.end_point_path_length)))

    # Get length of all neurites in cell by iterating over segments,
    # and summing the segment lengths.
    # This should yield the same result as iterating over sections.
    print('Total neurite length:',
          sum(iter_neurites(nrn, seg.length)))

    # get volume of all neurites in cell by summing over segment
    # volumes
    print('Total neurite volume:',
          sum(iter_neurites(nrn, seg.volume)))

    # get area of all neurites in cell by summing over segment
    # areas
    print('Total neurite surface area:',
          sum(iter_neurites(nrn, seg.area)))

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
          np.mean(list(iter_neurites(nrn, seg.radius))))

    # get stats for the segment taper rate, for different types of neurite
    for ttype in ezy.NEURITE_TYPES:
        ttt = ttype
        seg_taper_rate = list(iter_neurites(nrn, seg.taper_rate, tree_type_checker(ttt)))
        print('Segment taper rate (', ttype,
              '):\n  mean=', np.mean(seg_taper_rate),
              ', std=', np.std(seg_taper_rate),
              ', min=', np.min(seg_taper_rate),
              ', max=', np.max(seg_taper_rate),
              sep='')

    # Number of bifurcation points.
    print('Number of bifurcation points:', bif.count(nrn))

    # Number of bifurcation points for apical dendrites
    print('Number of bifurcation points (apical dendrites):',
          sum(1 for _ in iter_neurites(nrn, bif.identity,
                                       tree_type_checker(ezy.TreeType.apical_dendrite))))

    # Maximum branch order
    # This is complicated and will be factored into a helper function.
    # We iterate over sections, calcumating the branch order for each one.
    # The reason we cannot simply call nen.iter_sections(mt.branch_order) is
    # that mt.branch_order requires sections of tree nodes for navigation, but
    # nrn.iter_sections iterates over the sections of points.
    # TODO: This whole tree data business has to be refactored and simplified.
    print('Maximum branch order:',
          np.max([bo for bo in nrn.iter_neurites(
              lambda t: imap(mt.branch_order, isection(t)))]))

    # Neuron's bounding box
    print('Bounding box ((min x, y, z), (max x, y, z))',
          nrn.bounding_box())
