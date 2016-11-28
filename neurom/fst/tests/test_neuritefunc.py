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

from nose import tools as nt
import os
import numpy as np
import neurom as nm
from neurom.geom import convex_hull
from neurom.fst import _neuritefunc as _nf
from neurom.fst.sectionfunc import section_volume
from neurom.core import tree as tr
from neurom.core import Section, Neurite, Population

_PWD = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(_PWD, '../../../test_data/h5/v1/')
DATA_PATH = os.path.join(H5_PATH, 'Neuron.h5')
SWC_PATH = os.path.join(_PWD, '../../../test_data/swc')

NRN = nm.load_neuron(DATA_PATH)


def _equal(a, b, debug=False):
    if debug:
        print('\na.shape: %s\nb.shape: %s\n' % (a.shape, b.shape))
        print('\na: %s\nb:%s\n' % (a, b))
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.alltrue(a == b))


def _close(a, b, debug=False):
    if debug:
        print('\na.shape: %s\nb.shape: %s\n' % (a.shape, b.shape))
        print('\na: %s\nb:%s\n' % (a, b))
        print('\na - b:%s\n' % (a - b))
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.allclose(a, b))


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


def test_section_radial_distances_displaced_neurite():
    nrns = [nm.load_neuron(os.path.join(SWC_PATH, f)) for
            f in ('point_soma_single_neurite.swc', 'point_soma_single_neurite2.swc')]

    pop = Population(nrns)

    rad_dist_nrns = []
    for nrn in nrns:
        rad_dist_nrns.extend( nm.get('section_radial_distances', nrn))

    rad_dist_nrns = np.array(rad_dist_nrns)

    rad_dist_pop = nm.get('section_radial_distances', pop)

    nt.ok_(np.alltrue(rad_dist_pop == rad_dist_nrns))



def test_segment_radial_distances_displaced_neurite():
    nrns = [nm.load_neuron(os.path.join(SWC_PATH, f)) for
            f in ('point_soma_single_neurite.swc', 'point_soma_single_neurite2.swc')]

    pop = Population(nrns)

    rad_dist_nrns = []
    for nrn in nrns:
        rad_dist_nrns.extend( nm.get('segment_radial_distances', nrn))

    rad_dist_nrns = np.array(rad_dist_nrns)

    rad_dist_pop = nm.get('segment_radial_distances', pop)

    nt.ok_(np.alltrue(rad_dist_pop == rad_dist_nrns))


def test_total_volume_per_neurite():

    vol = _nf.total_volume_per_neurite(NRN)
    nt.eq_(len(vol), 4)

    # calculate the volumes by hand and compare
    vol2 = []

    for n in NRN.neurites:
        vol2.append(sum(section_volume(s) for s in n.iter_sections()))

    nt.eq_(vol, vol2)

    # regression test
    ref_vol = [271.94122143951864, 281.24754646913954, 274.98039928781355, 276.73860261723024]
    nt.ok_(np.allclose(vol, ref_vol))


def test_volume_density_per_neurite():

    vol = np.array(_nf.total_volume_per_neurite(NRN))
    hull_vol = np.array([convex_hull(n).volume for n in nm.iter_neurites(NRN)])

    vol_density = _nf.volume_density_per_neurite(NRN)
    nt.eq_(len(vol_density), 4)
    nt.ok_(np.allclose(vol_density, vol / hull_vol))

    ref_density = [0.43756606998299519, 0.52464681266899216,
                   0.24068543213643726, 0.26289304906104355]
    nt.ok_(np.allclose(vol_density, ref_density))


def test_terminal_length_per_neurite():
    nrn = nm.load_neuron(os.path.join(SWC_PATH, 'simple.swc'))
    terminal_distances = np.array(_nf.terminal_path_lengths_per_neurite(nrn))
    np.testing.assert_allclose(terminal_distances,
                               np.array([5 + 5., 5 + 6., 4. + 6., 4. + 5]))
    terminal_distances = np.array(_nf.terminal_path_lengths_per_neurite(
        nrn, neurite_type=nm.AXON))
    np.testing.assert_allclose(terminal_distances,
                               np.array([4. + 6., 4. + 5.]))
