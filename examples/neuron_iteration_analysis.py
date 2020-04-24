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

"""Advanced analysis examples

These examples highlight more advanced neurom
morphometrics functionality using iterators.

"""

from __future__ import print_function
from neurom.core.dataformat import COLS
import neurom as nm
from neurom import geom
from neurom.features import sectionfunc
from neurom.core import Tree
from neurom.core.types import tree_type_checker, NEURITES
from neurom import morphmath as mm
import numpy as np


if __name__ == '__main__':

    filename = 'test_data/swc/Neuron.swc'

    #  load a neuron from an SWC file
    nrn = nm.load_neuron(filename)

    # Some examples of what can be done using iteration
    # instead of pre-packaged functions that return lists.
    # The iterations give us a lot of flexibility: we can map
    # any function that takes a segment or section.

    # Get of all neurites in cell by iterating over sections,
    # and summing the section lengths
    def sec_len(sec):
        """Return the length of a section."""
        return mm.section_length(sec.points)

    print('Total neurite length (sections):',
          sum(sec_len(s) for s in nm.iter_sections(nrn)))

    # Get length of all neurites in cell by iterating over segments,
    # and summing the segment lengths.
    # This should yield the same result as iterating over sections.
    print('Total neurite length (segments):',
          sum(mm.segment_length(s) for s in nm.iter_segments(nrn)))

    # get volume of all neurites in cell by summing over segment
    # volumes
    print('Total neurite volume:',
          sum(mm.segment_volume(s) for s in nm.iter_segments(nrn)))

    # get area of all neurites in cell by summing over segment
    # areas
    print('Total neurite surface area:',
          sum(mm.segment_area(s) for s in nm.iter_segments(nrn)))

    # get total number of neurite points in cell.
    def n_points(sec):
        """number of points in a section."""
        n = len(sec.points)
        # Non-root sections have duplicate first point
        return n if sec.parent is None else n - 1

    print('Total number of points:',
          sum(n_points(s) for s in nm.iter_sections(nrn)))

    # get mean radius of neurite points in cell.
    # p[COLS.R] yields the radius for point p.
    # Note: this includes duplicated points at beginning of
    # non-trunk sections
    print('Mean radius of points:',
          np.mean([s.points[:, COLS.R] for s in nm.iter_sections(nrn)]))

    # get mean radius of neurite points in cell.
    # p[COLS.R] yields the radius for point p.
    # Note: this includes duplicated points at beginning of
    # non-trunk sections
    pts = [p[COLS.R] for s in nrn.sections[1:] for p in s.points]
    print('Mean radius of points:',
          np.mean(pts))

    # get mean radius of segments
    print('Mean radius of segments:',
          np.mean(list(mm.segment_radius(s) for s in nm.iter_segments(nrn))))

    # get stats for the segment taper rate, for different types of neurite
    for ttype in NEURITES:
        ttt = ttype
        seg_taper_rate = [mm.segment_taper_rate(s)
                          for s in nm.iter_segments(nrn, neurite_filter=tree_type_checker(ttt))]

        print('Segment taper rate (', ttype,
              '):\n  mean=', np.mean(seg_taper_rate),
              ', std=', np.std(seg_taper_rate),
              ', min=', np.min(seg_taper_rate),
              ', max=', np.max(seg_taper_rate),
              sep='')

    # Number of bifurcation points.
    print('Number of bifurcation points:',
          sum(1 for _ in nm.iter_sections(nrn,
                                          iterator_type=Tree.ibifurcation_point)))

    # Number of bifurcation points for apical dendrites
    print('Number of bifurcation points (apical dendrites):',
          sum(1 for _ in nm.iter_sections(nrn,
                                          iterator_type=Tree.ibifurcation_point,
                                          neurite_filter=tree_type_checker(nm.APICAL_DENDRITE))))

    # Maximum branch order
    print('Maximum branch order:',
          max(sectionfunc.branch_order(s) for s in nm.iter_sections(nrn)))

    # Neuron's bounding box
    # Note: does not account for soma radius
    print('Bounding box ((min x, y, z), (max x, y, z))', geom.bounding_box(nrn))
