# Copyright (c) 2020, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
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

"""Neuron features.

Any public function from this namespace can be called via features mechanism on a neuron, a
neuron population:

>>> import neurom
>>> from neurom import features
>>> nrn = neurom.load_neuron('path/to/neuron')
>>> features.get('soma_surface_area', nrn)
>>> nrn_population = neurom.load_neurons('path/to/neurons')
>>> features.get('sholl_frequency', nrn_population)
"""


from collections import Iterable
from functools import partial
import math

import numpy as np

from neurom import morphmath
from neurom.core.neuron import iter_neurites, iter_segments
from neurom.core.dataformat import COLS
from neurom.core.types import NeuriteType
from neurom.core.types import tree_type_checker as is_type
from neurom.features import feature, neuritefunc

feature = partial(feature, namespace='NEURONFEATURES')


def _assure_iterable(neurons):
    """Makes sure `neurons` is an iterable of neurons."""
    if hasattr(neurons, 'neurons'):
        return neurons.neurons
    elif isinstance(neurons, Iterable):
        return neurons
    else:
        return (neurons,)


def _iter_neurites(neurons, neurite_type):
    neurite_filter = is_type(neurite_type)
    for n in neurons:
        for s in n.neurites:
            if neurite_filter(s):
                yield s, n


@feature(shape=())
def soma_volume(neuron):
    """Get the volume of a neuron's soma."""
    return neuron.soma.volume


@feature(shape=(...,))
def soma_volumes(neurons):
    """Get the volume of the somata in a population of neurons.

    Note:
        If a single neuron is passed, a single element list with the volume
        of its soma member is returned.
    """
    return [soma_volume(n) for n in _assure_iterable(neurons)]


@feature(shape=())
def soma_surface_area(neuron, neurite_type=NeuriteType.soma):
    """Get the surface area of a neuron's soma.

    Note:
        The surface area is calculated by assuming the soma is spherical.
    """
    assert neurite_type == NeuriteType.soma, 'Neurite type must be soma'
    return 4 * math.pi * neuron.soma.radius ** 2


@feature(shape=(...,))
def soma_surface_areas(neurons, neurite_type=NeuriteType.soma):
    """Get soma surface areas of of neurons.

    Note:
        The surface area is calculated by assuming the soma is spherical.
    """
    assert neurite_type == NeuriteType.soma, 'Neurite type must be soma'
    return [soma_surface_area(n) for n in _assure_iterable(neurons)]


@feature(shape=(...,))
def soma_radii(neurons, neurite_type=NeuriteType.soma):
    """Get the radii of the somata of a population of neurons."""
    assert neurite_type == NeuriteType.soma, 'Neurite type must be soma'
    return [n.soma.radius for n in _assure_iterable(neurons)]


@feature(shape=(...,))
def trunk_section_lengths(neurons, neurite_type=NeuriteType.all):
    """List of lengths of trunk sections of neurites in a neuron."""
    neurons = _assure_iterable(neurons)
    return [morphmath.section_length(s.root_node.points)
            for s, n in _iter_neurites(neurons, neurite_type)]


@feature(shape=(...,))
def trunk_origin_radii(neurons, neurite_type=NeuriteType.all):
    """Radii of the trunk sections of neurites in a neuron."""
    neurons = _assure_iterable(neurons)
    return [s.root_node.points[0][COLS.R]
            for s, n in _iter_neurites(neurons, neurite_type)]


@feature(shape=(...,))
def trunk_origin_azimuths(neurons, neurite_type=NeuriteType.all):
    """Get a list of all the trunk origin azimuths of a neuron or population.

    The azimuth is defined as Angle between x-axis and the vector
    defined by (initial tree point - soma center) on the x-z plane.

    The range of the azimuth angle [-pi, pi] radians
    """
    neurons = _assure_iterable(neurons)

    def _azimuth(section, soma):
        """Azimuth of a section."""
        vector = morphmath.vector(section[0], soma.center)
        return np.arctan2(vector[COLS.Z], vector[COLS.X])

    return [_azimuth(s.root_node.points, n.soma)
            for s, n in _iter_neurites(neurons, neurite_type)]


@feature(shape=(...,))
def trunk_origin_elevations(neurons, neurite_type=NeuriteType.all):
    """Get a list of all the trunk origin elevations of a neuron or population.

    The elevation is defined as the angle between x-axis and the
    vector defined by (initial tree point - soma center)
    on the x-y half-plane.

    The range of the elevation angle [-pi/2, pi/2] radians
    """
    neurons = _assure_iterable(neurons)

    def _elevation(section, soma):
        """Elevation of a section."""
        vector = morphmath.vector(section[0], soma.center)
        norm_vector = np.linalg.norm(vector)

        if norm_vector >= np.finfo(type(norm_vector)).eps:
            return np.arcsin(vector[COLS.Y] / norm_vector)
        raise ValueError("Norm of vector between soma center and section is almost zero.")

    return [_elevation(s.root_node.points, n.soma)
            for s, n in _iter_neurites(neurons, neurite_type)]


@feature(shape=(...,))
def trunk_vectors(neurons, neurite_type=NeuriteType.all):
    """Calculates the vectors between all the trunks of the neuron and the soma center."""
    neurons = _assure_iterable(neurons)
    return np.array([morphmath.vector(s.root_node.points[0], n.soma.center)
                     for s, n in _iter_neurites(neurons, neurite_type)])


@feature(shape=(...,))
def trunk_angles(neurons, neurite_type=NeuriteType.all):
    """Calculates the angles between all the trunks of the neuron.

    The angles are defined on the x-y plane and the trees
    are sorted from the y axis and anticlock-wise.
    """
    vectors = trunk_vectors(neurons, neurite_type=neurite_type)
    # In order to avoid the failure of the process in case the neurite_type does not exist
    if not vectors.size:
        return []

    def _sort_angle(p1, p2):
        """Angle between p1-p2 to sort vectors."""
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return ang1 - ang2

    # Sorting angles according to x-y plane
    order = np.argsort(np.array([_sort_angle(i / np.linalg.norm(i), [0, 1])
                                 for i in vectors[:, 0:2]]))

    ordered_vectors = vectors[order][:, [COLS.X, COLS.Y]]

    return [morphmath.angle_between_vectors(ordered_vectors[i], ordered_vectors[i - 1])
            for i, _ in enumerate(ordered_vectors)]


def sholl_crossings(neurites, center, radii, neurite_type=NeuriteType.all):
    """Calculate crossings of neurites. The only function in this module that is not a feature.

    Args:
        neurites(list): morphology on which to perform Sholl analysis, or list of neurites
        center(Point): center point
        radii(iterable of floats): radii for which crossings will be counted
        neurite_type(NeuriteType): Type of neurite to use. By default ``NeuriteType.all`` is used.

    Returns:
        Array of same length as radii, with a count of the number of crossings
        for the respective radius

    This function can also be used with a list of sections, as follow::

        secs = (sec for sec in nm.iter_sections(neuron) if complex_filter(sec))
        sholl = nm.features.neuritefunc.sholl_crossings(secs,
                                                        center=neuron.soma.center,
                                                        radii=np.arange(0, 1000, 100))
    """
    def _count_crossings(neurite, radius):
        """Used to count_crossings of segments in neurite with radius."""
        r2 = radius ** 2
        count = 0
        for start, end in iter_segments(neurite):
            start_dist2, end_dist2 = (morphmath.point_dist2(center, start),
                                      morphmath.point_dist2(center, end))

            count += int(start_dist2 <= r2 <= end_dist2 or
                         end_dist2 <= r2 <= start_dist2)

        return count

    return np.array([sum(_count_crossings(neurite, r)
                         for neurite in iter_neurites(neurites, filt=is_type(neurite_type)))
                     for r in radii])


@feature(shape=(...,))
def sholl_frequency(neurons, neurite_type=NeuriteType.all, step_size=10, bins=None):
    """Perform Sholl frequency calculations on a population of neurites.

    Args:
        neurons(morph): neuron, list of neurons or neuron population
        neurite_type(NeuriteType): which neurites to operate on
        step_size(float): step size between Sholl radii
        bins(iterable of floats): custom binning to use for the Sholl radii. If None, it uses
        intervals of step_size between min and max radii of ``neurons``.

    Note:
        Given a neuron, the soma center is used for the concentric circles,
        which range from the soma radii, and the maximum radial distance
        in steps of `step_size`.  When a population is given, the concentric
        circles range from the smallest soma radius to the largest radial neurite
        distance.  Finally, each segment of the neuron is tested, so a neurite that
        bends back on itself, and crosses the same Sholl radius will get counted as
        having crossed multiple times.
    """
    neurons = _assure_iterable(neurons)
    neurite_filter = is_type(neurite_type)

    if bins is None:
        min_soma_edge = min(n.soma.radius for n in neurons)
        max_radii = max(np.max(np.linalg.norm(s.points[:, COLS.XYZ], axis=1))
                        for n in neurons
                        for s in n.neurites if neurite_filter(s))
        bins = np.arange(min_soma_edge, min_soma_edge + max_radii, step_size)

    return sum(sholl_crossings(n, n.soma.center, bins, neurite_type)
               for n in neurons)


@feature(shape=(...,))
def total_length(neurons, neurite_type=NeuriteType.all):
    """Get the total length of all sections in the group of neurons or neurites."""
    neurons = _assure_iterable(neurons)
    return list(sum(neuritefunc.section_lengths(n, neurite_type=neurite_type)) for n in neurons)


@feature(shape=(...,))
def max_radial_distances(neurons, neurite_type=NeuriteType.all):
    """Get the maximum radial distances of the termination sections.

    For a collection of neurites, neurons or a neuron population.
    """
    neurons = _assure_iterable(neurons)
    return [neuritefunc.max_radial_distance(n, neurite_type) for n in neurons]


@feature(shape=(...,))
def number_of_sections(neurons, neurite_type=NeuriteType.all):
    """Number of sections in a collection of neurites, neurons or a neuron population."""
    neurons = _assure_iterable(neurons)
    return [neuritefunc.n_sections(n, neurite_type) for n in neurons]


@feature(shape=(...,))
def number_of_neurites(neurons, neurite_type=NeuriteType.all):
    """Number of neurites in a collection of neurites, neurons or a neuron population."""
    neurons = _assure_iterable(neurons)
    return [neuritefunc.n_neurites(n, neurite_type) for n in neurons]


@feature(shape=(...,))
def number_of_bifurcations(neurons, neurite_type=NeuriteType.all):
    """Number of bifurcation points in a collection of neurites, neurons or a neuron population."""
    neurons = _assure_iterable(neurons)
    return [neuritefunc.n_bifurcation_points(n, neurite_type) for n in neurons]


@feature(shape=(...,))
def number_of_forking_points(neurons, neurite_type=NeuriteType.all):
    """Number of forking points in a collection of neurites, neurons or a neuron population."""
    neurons = _assure_iterable(neurons)
    return [neuritefunc.n_forking_points(n, neurite_type) for n in neurons]


@feature(shape=(...,))
def number_of_terminations(neurons, neurite_type=NeuriteType.all):
    """Number of leaves points in a collection of neurites, neurons or a neuron population."""
    neurons = _assure_iterable(neurons)
    return [neuritefunc.n_leaves(n, neurite_type) for n in neurons]


@feature(shape=(...,))
def number_of_segments(neurons, neurite_type=NeuriteType.all):
    """Number of sections in a collection of neurites, neurons or a neuron population."""
    neurons = _assure_iterable(neurons)
    return [neuritefunc.n_segments(n, neurite_type) for n in neurons]
