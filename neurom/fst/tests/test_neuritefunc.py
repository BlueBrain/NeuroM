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

'''Test neurom._neuritefunc functionality'''

from math import sqrt, pi
from nose import tools as nt
import os
import numpy as np
from numpy.testing import assert_allclose
import neurom as nm
from neurom.geom import convex_hull
from neurom.fst import _neuritefunc as _nf
from neurom.fst.sectionfunc import section_volume
from neurom.core import tree as tr
from neurom.core import Section, Neurite, Population

from utils import _close, _equal

_PWD = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(_PWD, '../../../test_data/h5/v1/')
DATA_PATH = os.path.join(H5_PATH, 'Neuron.h5')
SWC_PATH = os.path.join(_PWD, '../../../test_data/swc')
SIMPLE = nm.load_neuron(os.path.join(SWC_PATH, 'simple.swc'))
NRN = nm.load_neuron(DATA_PATH)


def test_principal_direction_extents():
    # test with a realistic neuron
    nrn = nm.load_neuron(os.path.join(H5_PATH, 'bio_neuron-000.h5'))

    p_ref = [1672.9694359427331, 142.43704397865031, 226.45895382204986,
             415.50612748523838, 429.83008974193206, 165.95410536922873,
             346.83281498399697]

    p = _nf.principal_direction_extents(nrn)
    _close(np.array(p), np.array(p_ref))


s0 = Section(42)
s1 = s0.add_child(Section(42))
s2 = s0.add_child(Section(42))
s3 = s0.add_child(Section(42))
s4 = s1.add_child(Section(42))
s5 = s1.add_child(Section(42))
s6 = s4.add_child(Section(42))
s7 = s4.add_child(Section(42))


def test_n_bifurcation_points():
    nt.assert_equal(_nf.n_bifurcation_points(Neurite(s0)), 2)
    nt.assert_equal(_nf.n_bifurcation_points(Neurite(s1)), 2)
    nt.assert_equal(_nf.n_bifurcation_points(Neurite(s2)), 0)
    nt.assert_equal(_nf.n_bifurcation_points(Neurite(s3)), 0)
    nt.assert_equal(_nf.n_bifurcation_points(Neurite(s4)), 1)
    nt.assert_equal(_nf.n_bifurcation_points(Neurite(s5)), 0)
    nt.assert_equal(_nf.n_bifurcation_points(Neurite(s6)), 0)
    nt.assert_equal(_nf.n_bifurcation_points(Neurite(s7)), 0)


def test_n_forking_points():
    nt.assert_equal(_nf.n_forking_points(Neurite(s0)), 3)
    nt.assert_equal(_nf.n_forking_points(Neurite(s1)), 2)
    nt.assert_equal(_nf.n_forking_points(Neurite(s2)), 0)
    nt.assert_equal(_nf.n_forking_points(Neurite(s3)), 0)
    nt.assert_equal(_nf.n_forking_points(Neurite(s4)), 1)
    nt.assert_equal(_nf.n_forking_points(Neurite(s5)), 0)
    nt.assert_equal(_nf.n_forking_points(Neurite(s6)), 0)
    nt.assert_equal(_nf.n_forking_points(Neurite(s7)), 0)


def test_n_leaves():
    nt.assert_equal(_nf.n_leaves(Neurite(s0)), 5)
    nt.assert_equal(_nf.n_leaves(Neurite(s1)), 3)
    nt.assert_equal(_nf.n_leaves(Neurite(s2)), 1)
    nt.assert_equal(_nf.n_leaves(Neurite(s3)), 1)
    nt.assert_equal(_nf.n_leaves(Neurite(s4)), 2)
    nt.assert_equal(_nf.n_leaves(Neurite(s5)), 1)
    nt.assert_equal(_nf.n_leaves(Neurite(s6)), 1)
    nt.assert_equal(_nf.n_leaves(Neurite(s7)), 1)


def test_total_volume_per_neurite():
    vol = _nf.total_volume_per_neurite(NRN)
    nt.eq_(len(vol), 4)

    # calculate the volumes by hand and compare
    vol2 = [sum(section_volume(s) for s in n.iter_sections())
			for n in NRN.neurites
            ]
    nt.eq_(vol, vol2)

    # regression test
    ref_vol = [271.94122143951864, 281.24754646913954,
               274.98039928781355, 276.73860261723024]
    nt.ok_(np.allclose(vol, ref_vol))


def test_volume_density_per_neurite():

    vol = np.array(_nf.total_volume_per_neurite(NRN))
    hull_vol = np.array([convex_hull(n).volume for n in nm.iter_neurites(NRN)])

    vol_density = _nf.volume_density_per_neurite(NRN)
    nt.eq_(len(vol_density), 4)
    nt.ok_(np.allclose(vol_density, vol / hull_vol))

    ref_density = [0.43756606998299519, 0.52464681266899216,
                   0.24068543213643726, 0.26289304906104355]
    assert_allclose(vol_density, ref_density)


def test_terminal_length_per_neurite():
    terminal_distances = _nf.terminal_path_lengths_per_neurite(SIMPLE)
    assert_allclose(terminal_distances,
                    (5 + 5., 5 + 6., 4. + 6., 4. + 5))
    terminal_distances = _nf.terminal_path_lengths_per_neurite(SIMPLE,
                                                               neurite_type=nm.AXON)
    assert_allclose(terminal_distances,
                    (4. + 6., 4. + 5.))

def test_total_length_per_neurite():
    total_lengths = _nf.total_length_per_neurite(SIMPLE)
    assert_allclose(total_lengths,
                    (5. + 5. + 6., 4. + 5. + 6.))

def test_n_segments():
    n_segments = _nf.n_segments(SIMPLE)
    nt.eq_(n_segments, 6)

def test_n_neurites():
    n_neurites = _nf.n_neurites(SIMPLE)
    nt.eq_(n_neurites, 2)

def test_n_sections():
    n_sections = _nf.n_sections(SIMPLE)
    nt.eq_(n_sections, 6)

def test_neurite_volumes():
    #note: cannot use SIMPLE since it lies in a plane
    total_volumes = _nf.total_volume_per_neurite(NRN)
    assert_allclose(total_volumes,
                    [271.94122143951864, 281.24754646913954,
                     274.98039928781355, 276.73860261723024]
                    )

def test_section_path_lengths():
    path_lengths = list(_nf.section_path_lengths(SIMPLE))
    assert_allclose(path_lengths,
                    (5., 10., 11., # type 3, basal dendrite
                     4., 10., 9.)) # type 2, axon

def test_n_sections_per_neurite():
    sections = _nf.n_sections_per_neurite(SIMPLE)
    assert_allclose(sections,
                    (3, 3))

def test_section_branch_orders():
    branch_orders = list(_nf.section_branch_orders(SIMPLE))
    assert_allclose(branch_orders,
                    (0, 1, 1,  # type 3, basal dendrite
                     0, 1, 1)) # type 2, axon

def test_section_radial_distances():
    radial_distances = _nf.section_radial_distances(SIMPLE)
    assert_allclose(radial_distances,
                    (5.0, sqrt(5**2 + 5**2), sqrt(6**2 + 5**2),   # type 3, basal dendrite
                     4.0, sqrt(6**2 + 4**2), sqrt(5**2 + 4**2)))  # type 2, axon

def test_local_bifurcation_angles():
    local_bif_angles = list(_nf.local_bifurcation_angles(SIMPLE))
    assert_allclose(local_bif_angles,
                    (pi, pi))

def test_remote_bifurcation_angles():
    remote_bif_angles = list(_nf.remote_bifurcation_angles(SIMPLE))
    assert_allclose(remote_bif_angles,
                    (pi, pi))

def test_partition():
    partition = list(_nf.bifurcation_partitions(SIMPLE))
    assert_allclose(partition,
                    (1.0, 1.0))

def test_segment_lengths():
    segment_lengths = _nf.segment_lengths(SIMPLE)
    assert_allclose(segment_lengths,
                    (5.0, 5.0, 6.0,   # type 3, basal dendrite
                     4.0, 6.0, 5.0))  # type 2, axon

def test_segment_midpoints():
    midpoints = np.array(_nf.segment_midpoints(SIMPLE))
    assert_allclose(midpoints,
                    np.array([[ 0. ,  (5. + 0) / 2,  0. ],  #trunk type 2
                              [-2.5,  5. ,  0. ],
                              [ 3. ,  5. ,  0. ],
                              [ 0. , (-4. + 0)/ 2. ,  0. ],  #trunk type 3
                              [ 3. , -4. ,  0. ],
                              [-2.5, -4. ,  0. ]]))

def test_segment_radial_distances():
    '''midpoints on segments'''
    radial_distances = _nf.segment_radial_distances(SIMPLE)
    assert_allclose(radial_distances,
                    (2.5, sqrt(2.5**2 + 5**2), sqrt(3**2 + 5**2),
                     2.0, 5.0, sqrt(2.5**2 + 4**2)))

def test_principal_direction_extents():
    principal_dir = list(_nf.principal_direction_extents(SIMPLE))
    assert_allclose(principal_dir,
                    (14.736052694538641, 12.105102672688004))
